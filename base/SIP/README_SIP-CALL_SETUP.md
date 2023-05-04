## Installing baresip and adding the vidpipe module in baresip. [Linux and ARM64]
1. Inside thirdparty/baresip/modules create a new folder vidpipe.
2. In the vidpipe folder add the files vidpipe.c and corresponding CMakeList of vidpipe.
3. Edit the file baresip/cmake/modules.cmake and append "vidpipe" in set(MODULES).
4. Follow the build instructions in  https://github.com/baresip/baresip/wiki/Install:-Stable-Release and build lib_re and baresip.
5. Run baresip in home and verify build. The account, contacts, config files can be found by doing cd .baresip/ in home dir.

## SIP Call Demo [Linux and ARM64]
1. Edit the config file and add 'module vidpipe' in  #video source modules section. Uncomment the modules required for making a video call eg vidcodec, vidfilters etc. Save and exit.
2. In Aprapipes run the sip test cases. 
3. Baresip runs on command line, to make a call we need a sip account. Free Sip account can be created online.
4. The user agent can register in command line or by adding sip account in account file.
   Eg: In command line /uanew sip:vinapra@sip2sip.info
5. On phone download any free sip application and register a different sip account and configure the application settings to support video calls.
6. Calls can be made from the system to a phone and vice versa. 
7. To make a call to a sip account from the system 
   /dial sip:sipaccount.info
8. We can configure and specify udp/tcp by adding the tags while making calls.