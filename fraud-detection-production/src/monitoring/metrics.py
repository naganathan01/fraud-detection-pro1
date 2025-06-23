# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
import logging
from typing import Dict, Optional

# Request metrics
request_count = Counter(
    'fraud_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'fraud_api_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

# Model metrics
model_predictions = Counter(
    'fraud_model_predictions_total',
    'Total model predictions',
    ['model_version', 'result']
)

model_latency = Histogram(
    'fraud_model_latency_seconds',
    'Model prediction latency',
    ['model_name']
)

fraud_detection_rate = Gauge(
    'fraud_detection_rate',
    'Current fraud detection rate'
)

# Business metrics
transactions_processed = Counter(
    'transactions_processed_total',
    'Total transactions processed',
    ['result']
)

fraud_amount_prevented = Counter(
    'fraud_amount_prevented_total',
    'Total fraud amount prevented in USD'
)

# Feature store metrics
feature_cache_hits = Counter(
    'feature_cache_hits_total',
    'Feature store cache hits'
)

feature_cache_misses = Counter(
    'feature_cache_misses_total',  
    'Feature store cache misses'
)

class ModelMetrics:
    """Model-spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-detection-api
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: fraud_api_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60

---
# infrastructure/kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection-service
  namespace: fraud-detection
  labels:
    app: fraud-detection-api
spec:
  selector:
    app: fraud-detection-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP

---
# infrastructure/kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fraud-detection-ingress
  namespace: fraud-detection
  annotations:
    kubernetes.io/ingress.class: "alb"
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/load-balancer-name: fraud-detection-alb
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-west-2:123456789012:certificate/12345678-1234-1234-1234-123456789012
    alb.ingress.kubernetes.io/healthcheck-path: /health/ready
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: '30'
    alb.ingress.kubernetes.io/healthy-threshold-count: '2'
    alb.ingress.kubernetes.io/unhealthy-threshold-count: '3'
spec:
  rules:
  - host: fraud-api.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fraud-detection-service
            port:
              number: 80

---
# infrastructure/kubernetes/redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: fraud-detection
  labels:
    app: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command:
        - redis-server
        - --appendonly
        - "yes"
        - --maxmemory
        - "2gb"
        - --maxmemory-policy
        - "allkeys-lru"
        resources:
          requests:
            memory: "1Gi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "500m"
        volumeMounts:
        - name: redis-storage
          mountPath: /data
        livenessProbe:
          tcpSocket:
            port: 6379
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc

---
# infrastructure/kubernetes/redis-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: fraud-detection
  labels:
    app: redis
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP

---
# infrastructure/kubernetes/redis-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: fraud-detection
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: gp3

---
# infrastructure/kubernetes/monitoring.yaml
apiVersion: v1
kind: Service
metadata:
  name: prometheus-service
  namespace: fraud-detection
  labels:
    app: prometheus
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: fraud-detection
  labels:
    app: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-storage
          mountPath: /prometheus
        command:
        - /bin/prometheus
        - --config.file=/etc/prometheus/prometheus.yml
        - --storage.tsdb.path=/prometheus
        - --web.console.libraries=/usr/share/prometheus/console_libraries
        - --web.console.templates=/usr/share/prometheus/consoles
        - --storage.tsdb.retention.time=15d
        - --web.enable-lifecycle
        resources:
          requests:
            memory: "1Gi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "500m"
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
      - name: prometheus-storage
        persistentVolumeClaim:
          claimName: prometheus-pvc

---
# infrastructure/kubernetes/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: fraud-detection-network-policy
  namespace: fraud-detection
spec:
  podSelector:
    matchLabels:
      app: fraud-detection-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    ific metrics collection"""
    
    def __init__(self):
        self.prediction_times = []
        self.fraud_predictions = 0
        self.total_predictions = 0
        
    def record_prediction(self, latency: float, probability: float, is_fraud: bool):
        """Record prediction metrics"""
        # Update counters
        self.total_predictions += 1
        if is_fraud:
            self.fraud_predictions += 1
            
        # Record latency
        model_latency.labels(model_name='ensemble').observe(latency)
        
        # Record prediction
        result = 'fraud' if is_fraud else 'legitimate'
        model_predictions.labels(model_version='v2.0', result=result).inc()
        transactions_processed.labels(result=result).inc()
        
        # Update fraud detection rate
        if self.total_predictions > 0:
            rate = self.fraud_predictions / self.total_predictions
            fraud_detection_rate.set(rate)
            
    def record_error(self, error_type: str):
        """Record prediction errors"""
        model_predictions.labels(model_version='v2.0', result='error').inc()
        logging.error(f"Model prediction error: {error_type}")
        
    def record_business_impact(self, transaction_amount: float, is_fraud_prevented: bool):
        """Record business impact metrics"""
        if is_fraud_prevented:
            fraud_amount_prevented.inc(transaction_amount)