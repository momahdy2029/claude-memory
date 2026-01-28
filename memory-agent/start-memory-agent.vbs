' Start Memory Agent silently in background
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

agentDir = "C:\Users\moham\Desktop\Claude Memory\memory-agent"
pythonCmd = "python """ & agentDir & "\main.py"""

WshShell.CurrentDirectory = agentDir
WshShell.Run "cmd /c " & pythonCmd, 0, False
