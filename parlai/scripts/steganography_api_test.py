from steganography_api import reset, setup, create_agent, post_secret, has_pending_message, send_stegotext, receive_stegotext

import os

print("Starting test")

settings_file = os.path.join(os.path.dirname(__file__), 'steganography_api_test_settings.csv')
cleartext = b'The password is secret'

# Send (on one device)
setup(open(settings_file, 'r'))
my_agent = create_agent(True)
other_agent = create_agent(False)

secret_index = post_secret(my_agent, cleartext)

stegotexts = []

while(has_pending_message(my_agent)):
    stegotexts.append(send_stegotext(my_agent))

reset()

# Receive (on another device)
setup(open(settings_file, 'r'))
other_agent = create_agent(False)
my_agent = create_agent(True)

responses = []

for stegotext in stegotexts:
    responses.append(receive_stegotext(other_agent, stegotext))

success = responses[-1] == cleartext
print("Test succeeded!" if success else "Test failed!")
assert(success)
