
-- capture.lua
-- Capture only. Radar and DCA1000 must already be configured.
-- Current setup: 64 frames, 1000 ms frame periodicity.
-- 3-TX TDM: chirps 0, 1, 2; 4 RX; 256 ADC samples.

local output_dir =
    "C:\\Users\\guest0\\Documents\\Github\\Radar\\outputs"

local capture_count = 1
local capture_ms = 9000
local expected_bytes = 8388608

local function file_size(path)
    local f = io.open(path, "rb")
    if not f then
        return -1
    end

    local size = f:seek("end")
    f:close()
    return size or -1
end

for i = 1, capture_count do
    local output_path = string.format(
        "%s\\cal_0deg_%03d.bin",
        output_dir,
        i
    )

    WriteToLog(
        "\n[STAP] Arming capture " .. i .. "\n",
        "green"
    )

    ar1.CaptureCardConfig_StartRecord(output_path, 1)
    RSTD.Sleep(2000)

    WriteToLog("[STAP] Starting frame...\n", "green")
    ar1.StartFrame()

    -- Allow the finite frame sequence to finish and data to drain.
    RSTD.Sleep(capture_ms)

    WriteToLog("[STAP] Stopping recording...\n", "green")
    ar1.CaptureCardConfig_StopRecord()
    RSTD.Sleep(1500)

    local size = file_size(output_path)

    if size >= 0 then
        WriteToLog(
            "[STAP] Capture " .. i .. " size was " .. size ..
            " bytes; expected " .. expected_bytes .. "\n",
            "red"
        )
    else
        WriteToLog(
            "[STAP] Could not check the supplied output filename. " ..
            "Inspect the actual DCA1000 output files.\n",
            "red"
        )
    end

    RSTD.Sleep(1000)
end

WriteToLog(
    "[STAP] Capture loop finished. Verify all three files and logs.\n",
    "green"
)
