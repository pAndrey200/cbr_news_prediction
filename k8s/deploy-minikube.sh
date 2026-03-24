#!/usr/bin/env bash
# deploy-minikube.sh — build and deploy cbr-news to local Minikube cluster
# Usage: bash k8s/deploy-minikube.sh [--rebuild]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
K8S_DIR="$PROJECT_ROOT/k8s"
NAMESPACE="cbr-news"
IMAGE="cbr-news:latest"

info()    { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
success() { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
die()     { echo -e "\033[1;31m[ERR]\033[0m   $*" >&2; exit 1; }


command -v minikube >/dev/null 2>&1 || die "minikube not found"
command -v kubectl  >/dev/null 2>&1 || die "kubectl not found"

if ! minikube status | grep -q "Running"; then
    info "Starting Minikube..."
    minikube start --memory=8192 --cpus=4
fi

CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"
DATA_DIR="$PROJECT_ROOT/data"

info "Mounting $CHECKPOINTS_DIR → /mnt/cbr-news/checkpoints (background)"
minikube ssh "sudo mkdir -p /mnt/cbr-news/checkpoints /mnt/cbr-news/data"
nohup minikube mount "$CHECKPOINTS_DIR:/mnt/cbr-news/checkpoints" > /tmp/mount-ckpt.log 2>&1 &
MOUNT_CKPT_PID=$!

nohup minikube mount "$DATA_DIR:/mnt/cbr-news/data" > /tmp/mount-data.log 2>&1 &
MOUNT_DATA_PID=$!

info "Mount PIDs: checkpoints=$MOUNT_CKPT_PID  data=$MOUNT_DATA_PID"
info "To stop mounts: kill $MOUNT_CKPT_PID $MOUNT_DATA_PID"
sleep 3 

if [[ "${1:-}" == "--rebuild" ]] || ! minikube image ls | grep -q "cbr-news:latest"; then
    info "Building $IMAGE on host Docker (uses layer cache)..."
    docker build -t "$IMAGE" "$PROJECT_ROOT"
    info "Loading $IMAGE into Minikube..."
    minikube image load "$IMAGE"
    success "Image loaded into Minikube: $IMAGE"
else
    info "Image $IMAGE already present, skipping build (use --rebuild to force)"
fi

# ── apply manifests ───────────────────────────────────────────────────────────
info "Applying Kubernetes manifests..."
kubectl apply -f "$K8S_DIR/namespace.yaml"
kubectl apply -f "$K8S_DIR/secrets.yaml"
kubectl apply -f "$K8S_DIR/configmap.yaml"
kubectl apply -f "$K8S_DIR/storage.yaml"
kubectl apply -f "$K8S_DIR/postgres.yaml"
kubectl apply -f "$K8S_DIR/redis.yaml"

info "Waiting for postgres to be ready..."
kubectl rollout status statefulset/postgres -n "$NAMESPACE" --timeout=120s
kubectl wait pod -l app=postgres -n "$NAMESPACE" --for=condition=ready --timeout=120s

kubectl apply -f "$K8S_DIR/api.yaml"
kubectl apply -f "$K8S_DIR/celery-worker.yaml"
kubectl apply -f "$K8S_DIR/bot.yaml"
kubectl apply -f "$K8S_DIR/cronjob-parser.yaml"
kubectl apply -f "$K8S_DIR/ingress.yaml"

success "All manifests applied."

echo ""
info "Cluster status:"
kubectl get pods -n "$NAMESPACE"
echo ""

API_URL=$(minikube service api -n "$NAMESPACE" --url 2>/dev/null || echo "run: minikube service api -n cbr-news --url")
success "API URL: $API_URL"
info "Tail logs:     kubectl logs -f deploy/api -n $NAMESPACE"
info "Manual parse:  kubectl create job --from=cronjob/cbr-parser parse-now -n $NAMESPACE"
