import boto3
import os


def extract_data():

    s3 = boto3.client('s3')
    bucket_name = 'stock-market-usecase'
    url = s3.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': 'datasets.zip'},
                    ExpiresIn=7200  # URL expiration time in seconds (adjust as needed)
                )
    print(url)
    return url

extract_data()
