# Code Style Guide & Framework Principles

Checklist for code style guide:

1. Have a `Detail` class inside the module class to enforce abstractions & data hiding.
2. No declarations in `.h` file - only base class methods overriding is allowed
3. Use mCamelCase in the module class
4. Format the code using `ctrl+shift+I`
5. No unused variables, unused header files
6. No commented code blocks
7. If any Detail class variable has to be accessed in module, don't use set, get - make the variables public
8. Avoid putting unneccessary headers in header files. Put the `#includes` in the `cpp` files as much as possible
9.  Module Properties:
   - No hardcoding
   - Dynamic support to change properties

Checklist for performance:
1. No memcopy
2. Initialize everything in `init`
3. Destroy everything in `term`
4. Use object pool for temporary variables (see `makeFrame`)
5. Unit Tests
6. Test for each supported color format (if applicable)
7. If output is image - use `saveorcompare` method
8. Always try to use `FileReader` module as input for transform modules
9. Use `step` rather than `pipeline.run_all_threaded()` in unit tests
10. Test for dynamic change of props and validation (`getProps`/`setProps`)
11. Use `ExternalSink` module wherever applicable
12. Unit test should not have sleep