import base64
import requests
import json

class IIMAPI():
	
	headers = {
		'Content-Type': 'application/json',
	}
	# change this
	# URI = "http://localhost:8000"
	URI = "mlst-iim-server.eurodyn.com:80"

	def __init__(self, username, password):
		print("[+] Initializing credentials")
		auth_str = ('%s:%s' % (username, password)).encode("utf-8")
		self.headers['Authorization'] = 'Basic %s' % (base64.b64encode(auth_str)).decode()
		
	def getDevices(self):
		API_URI = self.URI + "/devices/devices-api/"

		print("[+] GET Trying to connect")
		try:
			req = requests.get(API_URI, headers=self.headers)

		except requests.exceptions.RequestException as e:
			print("[-] Error..." + str(e))
			print("[-] Returning")
			return 0

		else:
			devices = req.json()
			print("[+] GET response: " + str(devices))

	def postDevices(self, data):

		API_URI = self.URI + "/devices/devices-api/"
		print("[+] POST Trying to connect")
		try:
			req = requests.post(API_URI, headers=self.headers, data=json.dumps(data))

		except requests.exceptions.RequestException as e:
			print("[-] Error..." + str(e))
			print("[-] Returning")
			return 0

		else:
			print("[+] POST response : " + str(req.content.decode()))


if __name__ == '__main__':

	# change this
	username = ''
	password = ''

	data = [
	  {
	  "node_id": "100",
	  "node_ip_address": "23.124.43.12",
	  "asset_location": "PPC-1",
	  "reputation_value": "55",
	  "reputation_change_speed": "44.6",
	  "timestamp": "",
	  "asset_value": "1"
	  }]
	
	api = VIDSDeviceAPI(username, password)
	api.getDevices()
	api.postDevices(data)


