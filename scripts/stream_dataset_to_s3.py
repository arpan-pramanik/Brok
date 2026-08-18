import os
import json
import urllib.request
import boto3

BUCKET_NAME = os.getenv("S3_BUCKET", "bragi-msmarco-xi-dataset")
REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

PARQUET_LIST_URL = "https://huggingface.co/api/datasets/ai4bharat/MSMARCO-XI/parquet"

def main():
    s3_client = boto3.client("s3", region_name=REGION)
    
    print(f"Fetching parquet manifest from Hugging Face ({PARQUET_LIST_URL})...")
    req = urllib.request.Request(PARQUET_LIST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        manifest = json.loads(resp.read().decode())

    default_split = manifest.get("default", {})
    
    total_files = 0
    for split_name, urls in default_split.items():
        for i, url in enumerate(urls):
            s3_key = f"msmarco-xi/{split_name}/{i}.parquet"
            print(f"[{total_files+1}] Streaming {split_name}/{i}.parquet directly to s3://{BUCKET_NAME}/{s3_key}...")
            
            req_file = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_file) as stream_resp:
                s3_client.upload_fileobj(
                    Fileobj=stream_resp,
                    Bucket=BUCKET_NAME,
                    Key=s3_key,
                    ExtraArgs={"ContentType": "application/x-parquet"}
                )
            print(f"✅ s3://{BUCKET_NAME}/{s3_key} uploaded!")
            total_files += 1

    print(f"\n🎉 Finished streaming all {total_files} dataset parquet files directly to AWS S3 bucket '{BUCKET_NAME}'!")

if __name__ == "__main__":
    main()
