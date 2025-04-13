# Swift-MLX TTS examples for macOS

This is an example of MLX-Swift running local TTS models.

Steps to run:

 - Open project in Xcode
 - Copy your .tensorfile to relevant Resources folder (Kokoro and Orpheus)
 - Change project signing in "Signing and Capabilities" project settings
 - Run the App
 
 nSpeak framework is embedeed for Kokoro already.

# Kokoro

Implemented and working. Based on [Kokoro TTS for iOS](https://github.com/mlalma/kokoro-ios).  All credit to mlalma for that work!

Uses MLX Swift and eSpeak NG.  M1 chip or better is requied.


# Orpheus

Test implementation - not working.  Unexpected layer shapes.
