import boto3

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print(f"Upload Successful: {s3_file}")
    except Exception as e:
        print(f"Upload Failed: {e}")

if __name__ == "__main__":
    local_model_path = 'models/saved_model'
    bucket_name = 'your-bucket-name'
    s3_model_path = 'models/saved_model'
    
    upload_to_aws(local_model_path, bucket_name, s3_model_path)
