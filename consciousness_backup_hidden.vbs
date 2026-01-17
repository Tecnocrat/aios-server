Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -ExecutionPolicy Bypass -File ""C:\dev\aios-server\consciousness_backup.ps1"" -Action backup", 0, False
