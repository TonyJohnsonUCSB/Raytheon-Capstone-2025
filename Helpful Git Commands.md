# Git Commands
(# is not the proper way to comment in cmd. It's only for reading convenience.)
## Editing the code...

1. Clone the repository on **your own computer** https://github.com/TonyJohnsonUCSB/Raytheon-Capstone-2025.git

2. Code on it

3. In cmd/powershell/git bash (find it by search on your computer), 

   ```cmd
   cd "C:\<your-directory>\Raytheon-Capstone-2025" # Find it by right click the folder in vscode and click "Copy Path"
   git add .
   git commit -m "Your descriptive commit message"
   git push origin branch-name # For branch-name you can use "main" or whatever new branches we'll have later on
   ```

4. Now it should be pushed up to Github. Check on the website.

5. If you want to download the newest version on Pi,

   ```cmd
   cd Raytheon-Capstone-2025
   sudo git pull # sudo is required on Pi. For your windows PC, remove sudo
   ```
   
## Running the code...
   ```cmd
   cd Raytheon-Capstone-2025
   source venv/bin/activate
   (sudo) python3 your-code.py # Use sudo if necessary (required for Pi communication)
   ```
