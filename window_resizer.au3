#include <Array.au3>

Local $stardewTitle = "Stardew Valley" ;Get the title of the Stardew Valley Process
;WinMove($stardewTitle, "", 10, 10, 640, 360) ;Move process to 10, 10 andresize to 640, 360 resolution
WinSetState($stardewTitle, "", @SW_MINIMIZE)
WinActivate($stardewTitle) ;Make Stardew Valley the active window
