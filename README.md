# COMP 472 Artificial intelligence 
## Instruction to Run Jupyter Notebook
- After you can successfully run conda from powershell, you can install jupyter with:
 `conda install jupyter` command. 
- Then re-open powershell and run `conda run jupyter notebook`.
- Another way: **Change ENV** ~/.zshrc
 ```zsh
  $ vim ./zshrc
  # press i/o to insert 
  >>> export PATH="/usr/local/Caskroom/miniconda/base/bin:$PATH"
  # press ESC to back to Command line in ./zshrc
  :wq #Save File
  #Success Run
  $ jupyter notebook
 ```
