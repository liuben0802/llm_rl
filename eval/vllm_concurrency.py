import asyncio
import time
import aiohttp
import statistics
import threading
import traceback
import json
from eval.rule_reward import rule_based_reward

class VLLMRequest:
    def __init__(self, url, modelname, concurrency, evalFunction):
        self.datas = []
        self.dataLock = threading.Lock()
        self.isConsumerTime = False
        self.isSaveTime = False
        self.resultData = []
        self.resultLock = threading.Lock()
        self.count = 0
        self.url = url
        self.model = modelname
        self.concurrency = concurrency
        self.total_requests = 1000
        self.max_tokens = 16384

    def push_result(self, data):
        self.resultLock.acquire()
        try:
            self.resultData.append(data)
        finally:
            self.resultLock.release()

    def pop_result(self):
        self.resultLock.acquire()
        try:
            data = self.resultData.pop(0)
        except:
            data = ""
        finally:
            self.resultLock.release()
        return data

    def getResultLen(self):
        self.resultLock.acquire()
        try:
            data = len(self.resultData)
        except:
            data = 0
        finally:
            self.resultLock.release()
        return data


    def push_data(self, data):
        self.count += 1
        self.dataLock.acquire()
        try:
            self.datas.append(data)
        finally:
            self.dataLock.release()

    def pop_data(self):
        self.dataLock.acquire()
        try:
            data = self.datas.pop(0)
        except:
            data = ""
        finally:
            self.dataLock.release()
        return data

    def getDataLen(self):
        self.dataLock.acquire()
        try:
            dataLen = len(self.datas)
        except:
            dataLen = 0
        finally:
            self.dataLock.release()
        return dataLen

    async def __one_request(
        self,
        session,
        api_url,
        model,
        max_tokens,
        req_id,
    ):
        rawdata = self.pop_data()
        if len(rawdata) == 0:
            return 0, 0
        PROMPT = rawdata["prompt"]
        user_id = rawdata["user_id"]
        region_id = rawdata["region_id"]
        promptTitle = rawdata["promptTitle"]
        products_90d = rawdata["products_90d"]
        products_lastyear = rawdata["products_lastyear"]
        products_3d = rawdata["products_3d"]
        candidate_pool = rawdata["candidate_pool"]
        if len(PROMPT) == 0:
            return 0, 0
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": ""},{"role": "user", "content": f"{promptTitle}\n{PROMPT}"}],
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "chat_template_kwargs": {"enable_thinking": False},
            #"response_format": {"type":"json_object"},
            "stream": False
        }
        start = time.time()
        async with session.post(api_url, json=payload) as resp:
            data = await resp.json()
        latency = time.time() - start
        output_text = data["choices"][0]["message"]["content"].strip()
        if len(output_text) < 10:
            self.push_data(rawdata)
            return 0,0
        usage = data.get("usage", {})
        output_tokens = usage.get("output_tokens", len(output_text.split()))
        print(
            f"[{req_id:03d}] "
            f"latency={latency:.2f}s | "
            f"tokens={output_tokens:4d} | "
            f"tok/s={output_tokens/latency:6.1f}"
            , flush=True
        )
        score = rule_based_reward(output_text, products_90d, products_lastyear, products_3d, candidate_pool)
        self.push_result({"promptTitle": promptTitle,
                          "prompt": PROMPT,
                          "response": output_text,
                          "user_id": user_id,
                          "region_id": region_id,
                          "products_90d": products_90d,
                          "products_lastyear": products_lastyear,
                          "products_3d": products_3d,
                          "candidate_pool": candidate_pool,
                          "score": score})
        return latency, output_tokens

    async def __run(
        self,
        api_url,
        model,
        concurrency,
        total_requests,
        max_tokens,
    ):
        connector = aiohttp.TCPConnector(limit=concurrency)
        timeout = aiohttp.ClientTimeout(total=12000)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        ) as session:
            sem = asyncio.Semaphore(concurrency)

            async def sem_task(i):
                async with sem:
                    return await self.__one_request(
                        session,
                        api_url,
                        model,
                        max_tokens,
                        i,
                    )
            while self.isConsumerTime or self.getDataLen() > 0:
                try:
                    tasks = []
                    t1 = time.time()
                    for i in range(total_requests):
                        tasks.append(asyncio.create_task(sem_task(i)))

                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    latencies, tokens = zip(*results)

                    total_tokens = sum(tokens)
                    total_time = sum(latencies)
                    if total_time == 0:
                        continue
                    t2 = time.time()
                    print("\n========== SUMMARY ==========")
                    print(f"Total requests     : {total_requests}")
                    print(f"Concurrency        : {concurrency}")
                    print(f"Max tokens/request : {max_tokens}")
                    print(f"Total tokens       : {total_tokens}")
                    print(f"Avg latency        : {statistics.mean(latencies):.2f}s")
                    print(f"P95 latency        : {statistics.quantiles(latencies, n=20)[18]:.2f}s")
                    print(f"Overall throughput : {total_tokens/total_time:.1f} tok/s")
                    print(f"total time:{(t2 - t1):.5f}s", flush=True)
                except Exception as e:
                    print(traceback.format_exc(), flush=True)

    async def __saveData(self):
        datas = []
        while self.isSaveTime or self.getResultLen() > 0:
            try:
                data = self.pop_result()
                if len(data) > 0:
                    datas.append(data)
                else:
                    time.sleep(1)
                    continue

                if len(datas) > 100:
                    with open(self.savePath, "a+") as wf:
                        for item in datas:
                            wf.write(json.dumps(item, ensure_ascii=False) + "\n")
                    datas.clear()
            except Exception as e:
                print(traceback.format_exc(), flush=True)

        if len(datas) > 0:
            with open(self.savePath, "a+") as wf:
                for item in datas:
                    wf.write(json.dumps(item, ensure_ascii=False) + "\n")

    def startConsumer(self):
        self.isSaveTime = True
        self.isConsumerTime = True
        asyncio.run(
            self.__run(
                self.url,
                self.model,
                self.concurrency,
                self.total_requests,
                self.max_tokens,
            )
        )
        self.isSaveTime = False

    def startSaver(self):
        asyncio.run(self.__saveData())

    def stopConsumer(self):
        self.isConsumerTime = False