import json  
import time  
import logging  
import os  
  
logger=logging.getLogger(__name__)  
METRICS_FILE=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"logs","metrics.jsonl")  
  
def _ensure_log_dir():  
    os.makedirs(os.path.dirname(METRICS_FILE),exist_ok=True)  
  
def log_request(role,model,latency_ms,tokens_generated,success,error=None):  
    _ensure_log_dir()  
    entry={"timestamp":time.strftime("%%Y-%%m-%%dT%%H:%%M:%%S"),"role":role,"model":model,"latency_ms":round(latency_ms,1),"tokens_generated":tokens_generated,"success":success}  
    if error:  
        entry["error"]=str(error)[:200]  
    try:  
        with open(METRICS_FILE,"a",encoding="utf-8") as f:  
            f.write(json.dumps(entry)+"\n")  
    except Exception as e:  
        logger.warning("Failed to write metrics: "+str(e))  
