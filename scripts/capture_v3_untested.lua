
-- capture.lua
-- Capture only. Radar and DCA1000 must already be configured in Studio
-- (StaticConfig, DataConfig, SensorConfig all Set, DCA1000 connected).
-- One recording per run. Same Studio calls as the hardware-verified version.

local output_dir =
    "C:\\Users\\guest0\\Documents\\Github\\Radar\\outputs"

-- Change these 3 lines before each run:
local file_name = "overlook_004.bin"  -- new name each run, so nothing is overwritten
local num_frames = 600                -- = Studio SensorConfig -> Frame -> No of Frames
local periodicity_ms = 50             -- = Studio SensorConfig -> Frame -> Periodicity

local output_path = output_dir .. "\\" .. file_name
local wait_ms = num_frames * periodicity_ms + 5000  -- all frames + 5 s for the data to drain

WriteToLog("\n[STAP] Arming capture: " .. output_path .. "\n", "green")
ar1.CaptureCardConfig_StartRecord(output_path, 1)
RSTD.Sleep(2000)

WriteToLog("[STAP] Starting " .. num_frames .. " frames x " .. periodicity_ms .. " ms...\n", "green")
ar1.StartFrame()
RSTD.Sleep(wait_ms)

WriteToLog("[STAP] Stopping recording...\n", "green")
ar1.CaptureCardConfig_StopRecord()
RSTD.Sleep(1500)

WriteToLog("[STAP] Done: " .. output_path .. "\n", "green")
