import os
import uuid
from datetime import datetime

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial

s3uri_train = 's3://sagemaker-demo-oay/xgb-classification/data/train/train.csv'
s3uri_validation = 's3://sagemaker-demo-oay/xgb-classification/data/validation/validation.csv'
s3uri_test = 's3://sagemaker-demo-oay/xgb-classification/data/test/test.csv'
BUCKET = "sagemaker-demo-oay"
PREFIX = "xgb-classification"
REGION_NAME = "eu-west-1"
framework_version = '1.2-2'
docker_image_name = sagemaker.image_uris.retrieve(framework='xgboost', region=REGION_NAME, version=framework_version)

# Workaround while versions are not updated in SM SDK
framework_version = '1.3-1'
docker_image_name = docker_image_name[:-5] + framework_version

hyperparams = {"max_depth": 5,
               "subsample": 0.8,
               "num_round": 600,
               "eta": 0.2,
               "gamma": 4,
               "min_child_weight": 6,
               "objective": 'binary:logistic',
               "verbosity": 0
               }


def main():
    sm_env = os.getenv("ENV_NAME", "DEV")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    boto_session = boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key,
                                 region_name=REGION_NAME)
    sm_boto_client = boto_session.client("sagemaker")
    sm_session = sagemaker.session.Session(boto_session=boto_session, sagemaker_client=sm_boto_client,
                                           default_bucket=BUCKET)
    sm_session.default_bucket()

    try:
        sm_execution_role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        sm_execution_role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20220922T083895')['Role']['Arn']

    s3_input_train = TrainingInput(s3_data=s3uri_train, content_type="csv")
    s3_input_validation = TrainingInput(s3_data=s3uri_validation, content_type="csv")

    try:
        xgb_experiment = Experiment.load(f"binary-classification-xgboost-{sm_env}",
                                         sagemaker_boto_client=sm_boto_client)
    except:
        xgb_experiment = Experiment.create(experiment_name=f"binary-classification-xgboost-{sm_env}",
                                           description="Demo sagemake xgboost ",
                                           sagemaker_boto_client=sm_boto_client)

    for eta in [0.2, 0.3, 0.4, 0.4]:
        create_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        trial = Trial.create(trial_name=f"xgb-train-trial-{create_date}-{str(uuid.uuid1())}",
                             experiment_name=xgb_experiment.experiment_name,
                             sagemaker_boto_client=sm_boto_client)

        xgb = sagemaker.estimator.Estimator(image_uri=docker_image_name,
                                            role=sm_execution_role,
                                            hyperparameters=hyperparams,
                                            instance_count=1,
                                            instance_type='ml.m4.xlarge',
                                            output_path=f's3://{BUCKET}/{PREFIX}/output',
                                            base_job_name='xgboost-binary-classification',
                                            sagemaker_session=sm_session)

        xgb.fit(inputs={
            'train': s3_input_train,
            'validation': s3_input_validation
        },
            experiment_config={
                'ExperimentName': xgb_experiment.experiment_name,
                'TrialName': trial.trial_name,
                'TrialComponentDisplayName': 'Training',
            },
            wait=False,
        )

        hyperparams["eta"] = eta


if __name__ == "__main__":
    main()
