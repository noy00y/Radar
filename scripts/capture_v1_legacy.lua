
-- capture.lua
-- Assumes radar and DCA1000 are already configured in the GUI.

local output_path =
    "C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\stap_test.bin"

WriteToLog("[STAP] Arming DCA1000 recording...\n", "green")
ar1.CaptureCardConfig_StartRecord(output_path, 1)

RSTD.Sleep(2000)

WriteToLog("[STAP] Starting radar frame...\n", "green")
ar1.StartFrame()

-- Current configuration: 64 frames, 1000 ms period.
RSTD.Sleep(70000)

WriteToLog("[STAP] Stopping recording...\n", "green")
ar1.CaptureCardConfig_StopRecord()

RSTD.Sleep(2000)

WriteToLog("[STAP] Capture attempt finished. Check the DCA1000 log.\n", "green")
