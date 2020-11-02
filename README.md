# What this is

## Conceptually

Whereas encryption means to hide secret information within gibberish, steganography hides it within innocent-looking media.

This project is an API for linguistic steganography, meaning it hides arbitrary-length secret messages within an innocent-looking conversation between two or more agents. Using the power of ParlAI's language models, the conversations can be made to look somewhat natural.

## Actually

A fork of Facebook Research's [ParlAI](https://github.com/facebookresearch/ParlAI) chatbot that exposes an API for encoding and decoding secret messages within conversations.

This fork is currently based off of ParlAI `v0.9.3` tag on Github.

# How to use

## Dependencies

This project requires all the same dependencies as the version of ParlAI it is based on, no more or less. I recommend checking the guide in their [README](https://github.com/facebookresearch/ParlAI/blob/v0.9.3/README.md) for installation and setup.

## API

You'll find the full API in `parlai/scripts/steganography_api.py`, and example usage in `parlai/scripts/steganography_api_test.py`.

# Security considerations

This project is a proof of concept, and is not designed to be secure in practise. The following list of security issues and considerations is not necessarily complete.

- The list of packets (cleartext bytes and the HIDDEN_STOP_SIGNAL) should be XOR'd with a one-time-pad. That is not yet implemented. This may not only betray secrecy, but also the contents of messages (though not trivially, as a random seed is used to setup the models).
- The output format of stegotext messages is not natural. Dictionary items (including `.` and `'`) are always separated by spaces, and all letters are lowercase.
- Timing of messages (including their relative order) is comletely left to the API's caller. This is an obvious detection risk.
- ParlAI may connect to the internet (e.g. to download model data), obviously compromising secrecy.
- The secrecy of the system is no greater than the quality of the language model used, relative to the language model used by the adversary.
