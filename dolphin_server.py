import cmd
from dolphin_local import DolphinModel
import readline  # Enables arrow key navigation and command history

class DolphinShell(cmd.Cmd):
    intro = 'Welcome to the Dolphin shell. Type help or ? to list commands.\n'
    prompt = 'dolphin> '
    
    def __init__(self):
        super().__init__()
        print("Loading model... (this may take a minute)")
        self.model = DolphinModel(
            model_path="./Dolphin3.0-Llama3.1-8B",
            load_in_8bit=True
        )
        print("\nModel loaded and ready!")
        
    def do_generate(self, prompt):
        """Generate text from a prompt: generate <your prompt>"""
        if not prompt:
            print("Please provide a prompt")
            return
            
        formatted_prompt = f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        try:
            response = self.model.generate(formatted_prompt)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error generating response: {e}")
    
    def do_quit(self, arg):
        """Exit the Dolphin shell"""
        print("Goodbye!")
        return True
        
    def do_EOF(self, arg):
        """Exit on Ctrl-D (EOF)"""
        print("\nGoodbye!")
        return True
    
    # Aliases for convenience
    do_exit = do_quit
    do_q = do_quit

if __name__ == '__main__':
    try:
        DolphinShell().cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")