# Source the shared .bashrc file if it exists.
if [ -r /.bashrc ] ; then . /.bashrc ; fi
#module load python3/3.6.2
#module swap python3/3.7.11
#module load cuda/8.0
#module load cudnn/v7.0-prod-cuda8
#module load ffmpeg/4.2.2
#module load numpy/1.13.1-python-3.6.2-openblas-0.2.20 
#module load pandas/0.20.3-python-3.6.2

## for Tesla V100 16 GB
#module swap python3/3.8.1
#module swap openblas/0.3.7
#module swap cuda/11.8   
#module swap cudnn/v8.6.0.163-prod-cuda-11.X  
#module swap ffmpeg/4.2.2   
#module swap numpy/1.18.1-python-3.8.1-openblas-0.3.7 
#module swap pandas/1.0.3-python-3.8.1
#module load scipy/1.4.1-python-3.8.1  



# for Tesla A100 PCIE 80GB
module swap binutils/2.34
module swap gcc/8.4.0

module swap python3/3.8.2 
module swap openblas/0.3.9 
module swap cuda/11.8 
module swap cudnn/v8.6.0.163-prod-cuda-11.X  
module swap ffmpeg/4.2.2  
module swap numpy/1.18.2-python-3.8.2-openblas-0.3.9 
module swap pandas/1.0.3-python-3.8.2 
module swap scipy/1.4.1-python-3.8.2
module load matplotlib/3.2.1-python-3.8.2 

# Place your own code within the if-fi below to
# avoid it being executed on logins via remote shell,
# remote exec, batch jobs and other non-interactive logins.

# Set up the bash environment if interactive login.
if tty -s ; then

  # Set the system prompt.
  PS1="\w\n\h(\u) $ "
  export PS1

  # Set up some user command aliases.
  alias h=history
  alias source=.

  # Confirm before removing, replacing or overwriting files.
  alias rm="rm -i"
  alias mv="mv -i"
  alias cp="cp -i"

fi
