import os, json, redis
r = redis.from_url(os.environ["REDIS_URL"])
for _ in range(2):
    r.rpush("ml:jobs", json.dumps({"text":"p->(q->p)","theory":"pl"}))
print("ml:jobs =", r.llen("ml:jobs"))
