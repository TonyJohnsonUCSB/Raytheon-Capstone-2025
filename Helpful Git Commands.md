# Git Commands

## Editing the code...

1. Clone the repository on **your own computer** https://github.com/TonyJohnsonUCSB/Raytheon-Capstone-2025.git

2. Code on it

3. In cmd/powershell (find it by search on your computer), 

   ```cmd
   cd /your-repository/Raytheon-Capstone-2025 # Typically it's in C:\Users\<username>\Raytheon-Capstone-2025
   git add .
   git commit -m "Your descriptive commit message"
   git push origin branch-name # For branch-name you can use "main" or whatever new branches we'll have
   ```

4. Now it should be pushed up to Github. Check on the website.

5. If you want to download the newest version on Pi,

   ```cmd
   cd Raytheon-Capstone-2025
   sudo git pull
   ```

