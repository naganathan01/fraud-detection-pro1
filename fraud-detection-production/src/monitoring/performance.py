# src/monitoring/performance.py
import time
import psutil
import logging
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    response_times: List[float]

class PerformanceMonitor:
    """Monitor system and application performance"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics_history: List[PerformanceMetrics] = []
        self.request_times: List[float] = []
        
    def collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network
            network = psutil.net_io_counters()
            
            # Connections (simplified)
            connections = len(psutil.net_connections())
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                active_connections=connections,
                response_times=self.request_times.copy()
            )
            
            # Store in history
            self._add_to_history(metrics)
            
            # Clear request times
            self.request_times.clear()
            
            return metrics
            
        except Exception as e:
            logging.error(f"Failed to collect system metrics: {e}")
            raise
    
    def record_request_time(self, response_time: float):
        """Record API request response time"""
        self.request_times.append(response_time)
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary over recent history"""
        
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'avg_disk_percent': sum(m.disk_percent for m in recent_metrics) / len(recent_metrics),
            'avg_connections': sum(m.active_connections for m in recent_metrics) / len(recent_metrics),
            'total_requests': sum(len(m.response_times) for m in recent_metrics),
            'avg_response_time': self._calculate_avg_response_time(recent_metrics),
            'p95_response_time': self._calculate_p95_response_time(recent_metrics)
        }
    
    def check_performance_alerts(self) -> List[Dict]:
        """Check for performance issues that require alerts"""
        
        alerts = []
        
        if not self.metrics_history:
            return alerts
        
        latest = self.metrics_history[-1]
        
        # CPU alert
        if latest.cpu_percent > 80:
            alerts.append({
                'type': 'high_cpu',
                'value': latest.cpu_percent,
                'threshold': 80,
                'severity': 'warning' if latest.cpu_percent < 90 else 'critical'
            })
        
        # Memory alert
        if latest.memory_percent > 85:
            alerts.append({
                'type': 'high_memory',
                'value': latest.memory_percent,
                'threshold': 85,
                'severity': 'warning' if latest.memory_percent < 95 else 'critical'
            })
        
        # Disk alert
        if latest.disk_percent > 90:
            alerts.append({
                'type': 'high_disk',
                'value': latest.disk_percent,
                'threshold': 90,
                'severity': 'critical'
            })
        
        # Response time alert
        if latest.response_times:
            avg_response = sum(latest.response_times) / len(latest.response_times)
            if avg_response > 1.0:  # 1 second
                alerts.append({
                    'type': 'slow_response',
                    'value': avg_response,
                    'threshold': 1.0,
                    'severity': 'warning'
                })
        
        return alerts
    
    def _add_to_history(self, metrics: PerformanceMetrics):
        """Add metrics to history with size limit"""
        
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > self.history_size:
            self.metrics_history = self.metrics_history[-self.history_size:]
    
    def _calculate_avg_response_time(self, metrics_list: List[PerformanceMetrics]) -> float:
        """Calculate average response time"""
        
        all_times = []
        for metrics in metrics_list:
            all_times.extend(metrics.response_times)
        
        return sum(all_times) / len(all_times) if all_times else 0.0
    
    def _calculate_p95_response_time(self, metrics_list: List[PerformanceMetrics]) -> float:
        """Calculate 95th percentile response time"""
        
        all_times = []
        for metrics in metrics_list:
            all_times.extend(metrics.response_times)
        
        if not all_times:
            return 0.0
        
        all_times.sort()
        index = int(0.95 * len(all_times))
        return all_times[index] if index < len(all_times) else all_times[-1]

# Global performance monitor instance
performance_monitor = PerformanceMonitor()