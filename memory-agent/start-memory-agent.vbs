' Start Memory Agent silently in background
' Uses relative path detection - works on any system after cloning

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get the directory where this script is located (auto-detection)
scriptPath = WScript.ScriptFullName
agentDir = fso.GetParentFolderName(scriptPath)

pythonCmd = "python """ & agentDir & "\main.py"""

WshShell.CurrentDirectory = agentDir
WshShell.Run "cmd /c " & pythonCmd, 0, False
