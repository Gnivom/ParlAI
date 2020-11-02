from steganography_api import reset, setup, create_agent, post_secret, has_pending_message, send_stegotext, receive_stegotext

print("Starting test")

seed = 12345
cleartext = b'The password is secret'

# Send (on one device)
setup(seed)
my_id = create_agent(True)
other_id = create_agent(False)

secret_index = post_secret(my_id, cleartext)

stegotexts = []

while(has_pending_message(my_id)):
    stegotexts.append(send_stegotext(my_id))

reset()

# Receive (on another device)
setup(seed)
other_id = create_agent(False)
my_id = create_agent(True)

responses = []

for stegotext in stegotexts:
    responses.append(receive_stegotext(other_id, stegotext))

success = responses[-1] == cleartext
print("Test succeeded!" if success else "Test failed!")
assert(success)
