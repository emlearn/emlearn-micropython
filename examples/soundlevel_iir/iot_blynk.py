

import time
import requests

class BlynkClient():
    """
    Ref:
    https://docs.blynk.io/en/blynk.cloud/device-https-api/upload-set-of-data-with-timestamps-api

    """
    
    def __init__(self,
            token,
            hostname='blynk.cloud',
            protocol='https',
        ):

        self._batch_url = protocol + '://' + hostname + '/external/api/batch/update?token=' + token

    def post_telemetry(self, values : list[dict[str, float]]):
        """
        Send multiple telemetry values.
        The 'time' key should be in Unix milliseconds.
        """
        stream_values = {}
        notime_values = {}

        # Blynk HTTP API currently cannot send multiple timestamped values on multiple streams
        # so where time is specified - we shuffle the data into being organized per-stream
        # data which is not timestamped is handled sent in a single request
        for datapoint in values:
            if 'time' in datapoint.keys():
                t = datapoint['time']
                for key, value in datapoint.items():
                    if key == 'time':
                        continue
                    if key not in stream_values.keys():
                        stream_values[key] = []
                    stream_values[key].append((t, value))
            else:
                notime_values.update(datapoint)

        if notime_values:
            self.post_multiple(notime_values)
    
        for key, datapoints in stream_values.items():
            self.post_timestamped(key, datapoints)

    def post_multiple(self, values : dict[str, str]):
        """
        Post multiple values at current time
        Each entry in values must have key being a pin name, and the associated value
        """
        args = '&'.join([ f'{k}={v}' for k, v in values.items() ])
        url = self._batch_url + '&' + args
        print('post-multiple', url)
        r = requests.get(url, timeout=5.0)
        print('post-multiple-done', r.status_code)
        print('post-multiple-back', r.content)
        #assert r.status_code == 200, (r.status_code, r.content)

    def post_timestamped(self, pin : str, values : list[tuple[int, float]]):
        """
        Post multiple values from different times, for 1 stream
        Each entry in values must be a tuple with (timestamp, value)
        """

        payload = values
        url = self._batch_url+f'&pin={pin}'
        r = requests.post(url, json=payload)
        assert r.status_code == 200, (r.status_code, r.content)




