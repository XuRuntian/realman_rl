from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
print("✅ Isaac Lab App started successfully")
app.close()
