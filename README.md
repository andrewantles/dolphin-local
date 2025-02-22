# Things I can think of right now that I want to remember

## Dependencies:
I use `pipenv install <package>` for just about everything.  
When that doesn't work, I use `pipenv run pip install <package>`

Install most of the dependencies directly from the Pipfile, except for:
- torch
- cuda

### torch
I go directly to the pytorch install website, and it gives me the `pip` command I need to install everything.  
Again, use `pipenv run pip install <package>` if needed.

### cuda
I go to the following URL and use the `pipenv run <command>` option:
- https://docs.nvidia.com/cuda/archive/12.6.0/cuda-quick-start-guide/index.html#pip-wheels-windows

## Running the model
### One-off
You can call the `claude-script.py` file for one-off inference, but it takes a while to load the model to the GPU each time.  
### Dolphin server
Another option is to run the `dolphin_server.py`. This option requires typing "generate" at the beginning of each prompt. E.g. `generate tell me a bedtime story`  
It has no system prompt, so sometimes need to add one to the beginning of the user prompt.  
It cannot remember context - each chat is it's own complete history.
It's just a local server running in Python in a terminal. 

## Roadmap for improvement:
* I keep getting this error. Will need to address at some point:
   * `The load_in_4bit and load_in_8bit arguments are deprecated and will be removed in the future versions. Please, pass a BitsAndBytesConfig object in quantization_config argument instead.`
* Need to implement a system prompt
* Need to implement rolling context, vs just individual messages
* Both of the above two bullets may require an HTTP Flask server type setup (which would be awesome anyway)
