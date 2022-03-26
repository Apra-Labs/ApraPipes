## Installing golang and setting up the signalling server.

1. Install [golang](https://golang.org/doc/install) as per your operating system.
2. ```cd base/webRTCcomponents```
3. ```go mod tidy``` to tidy up the mod ```go.mod``` and ```go.sum``` file and download dependencies.
4. ```cd goSignallingServer``` and to run the webRTCSignalling server ```go run .```.
5. Let the server run as process in some terminal.

## Running the webrtc demo

1. Connect a webcam device.
2. Make sure your LD_LIBRARY_PATH contains the outInstall/lib/x86_64-linux-gnu folder.
3. Run the test gstwebrtcsink_tests/gstwebrtctestrawrgbwcs.
4. The pipeline shall connect to the signalling server.
5. Open ```index.html``` and enter pipeline's peer id ```666``` to establish connection.
6. The video coming from the webRTCSink should play.