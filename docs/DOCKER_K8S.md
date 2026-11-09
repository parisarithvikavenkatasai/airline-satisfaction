# Docker & Kubernetes

```bash
# prerequisites: pipeline has been run (deployment/ populated)
docker build -t airline-satisfaction-api:1.0 .
docker run --rm -p 8000:8000 airline-satisfaction-api:1.0
curl http://localhost:8000/health

# compare container vs local predictions
CONTAINER_URL=http://localhost:8000 pytest tests/test_local_vs_container.py   # PowerShell: $env:CONTAINER_URL="http://localhost:8000"

# kubernetes (Docker Desktop / minikube)
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods
# NodePort 30080 (minikube: minikube service airline-satisfaction-api --url)
curl http://localhost:30080/health
```
For minikube load the local image first: `minikube image load airline-satisfaction-api:1.0`.
