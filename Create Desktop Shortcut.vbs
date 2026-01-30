' Create Desktop Shortcut for Claude Memory Manager
' Run this script once to create a shortcut on your desktop

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get paths
strDesktop = WshShell.SpecialFolders("Desktop")
strScriptPath = fso.GetParentFolderName(WScript.ScriptFullName)
strTargetPath = strScriptPath & "\memory-agent\venv\Scripts\pythonw.exe"
strArguments = """" & strScriptPath & "\memory-agent\manager.py"""
strWorkingDir = strScriptPath & "\memory-agent"
strShortcutPath = strDesktop & "\Claude Memory Manager.lnk"

' Create shortcut
Set oShortcut = WshShell.CreateShortcut(strShortcutPath)
oShortcut.TargetPath = strTargetPath
oShortcut.Arguments = strArguments
oShortcut.WorkingDirectory = strWorkingDir
oShortcut.Description = "Claude Memory Manager - Manage the Memory Agent server"
oShortcut.Save

MsgBox "Desktop shortcut created successfully!", vbInformation, "Claude Memory Manager"
