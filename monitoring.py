from prometheus_client import start_http_server, Summary, Counter, Gauge

# Metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total RAG requests')
RESPONSE_TIME = Summary('rag_response_time_seconds', 'Response time in seconds')
LATENCY = Gauge('rag_latency_seconds', 'Current latency')
TOKEN_COST = Gauge('rag_token_cost', 'Current token cost')

def record_request():
    """Increment request counter."""
    REQUEST_COUNT.inc()

def record_latency(latency):
    """Record response latency."""
    RESPONSE_TIME.observe(latency)
    LATENCY.set(latency)

def record_token_cost(cost):
    """Record token cost."""
    TOKEN_COST.set(cost)