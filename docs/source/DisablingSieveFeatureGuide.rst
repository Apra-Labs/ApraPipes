Disabling Sieve Feature Guide
=============================

Transform Modules, I/O Pins & Actual Frame flow
""""""""""""""""""""""""""""""""""""""""""""""""
The graph of an Apra Pipes pipeline has two components, modules and pins. The modules are the graph nodes and pins are the edges. Both of these combined together
define an apra pipes pipeline. Please note this is just the build phase.
The Transform modules in Apra Pipes were designed in a way to take input frames (based on the input pins defined while building the pipeline), 
do some processing using those  frames and spit out the output frames. The pins that are added by this Transform module are defined in as the ouputPins of the module.
However, the flow of the frames through the module is slightly different. The output frames from the Transform module are pushed into the same Frame Container that 
hosts the input pins. This way the upstream frames are passed to the downstream modules, with new frames being added along the way. 

Sieve
""""""

Disabling Sieve 
""""""""""""""""
- Why was it Required ?
- Syntax

Limitations with Disabling Sieve 
"""""""""""""""""""""""""""""""""

Implications of Disabling Sieve 
""""""""""""""""""""""""""""""""