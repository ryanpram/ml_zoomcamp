#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://192.168.1.34:9696/predict'


customer = {
    "job": "retired", "duration": 445, "poutcome": "success"
}


response = requests.post(url, json=customer).json()
response


if response['churn'] == True:
    print('sending promo email to %s with proba: %f' % ('xyz-123',response['churn_probability']))
else:
    print('DONT sending promo email to %s with proba:%f' % ('xyz-123',response['churn_probability']))






