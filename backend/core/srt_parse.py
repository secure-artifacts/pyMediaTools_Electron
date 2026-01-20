"""
SRT 解析模块 - 从 SW_GenSubTitle/SrtParse.py 移植
"""


def timecodeToMilliseconds(srt_time):
    """将时间码字符串转换为总毫秒数"""
    h, m, s_ms = srt_time.split(":")
    s, ms = s_ms.split(",")
    return (int(h) * 3600000) + (int(m) * 60000) + (int(s) * 1000) + int(ms)


def timecodeToSeconds(timecode):
    """将SRT时间码转换为总秒数"""
    return timecodeToMilliseconds(timecode) / 1000.0


def millisecondsToTimecode(ms):
    """将毫秒转换为SRT时间码格式"""
    hours = int(ms // 3600000)
    minutes = int((ms % 3600000) // 60000)
    seconds = int((ms % 60000) // 1000)
    milliseconds = int(ms % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def secondsToTimecode(seconds):
    """将总秒数转换为SRT时间码格式"""
    milliseconds = seconds * 1000
    return millisecondsToTimecode(milliseconds)


class SrtParse:
    """SRT字幕解析类"""
    
    def __init__(self, srtFile, ignore=""):
        self.srtInfos = []
        self.totaleCount = 0
        self.totalTime = 0
        self.parse(srtFile, ignore)

    def parse(self, srtFile, ignore):
        """解析SRT文件"""
        lines = []
        with open(srtFile, "r", encoding="utf8") as f:
            lines = f.readlines()
            
        count = len(lines)
        index = 0
        while index < count:
            line = lines[index]
            index += 1
            if line.strip() == "":
                continue

            # 序号
            try:
                number = int(line.strip())
            except:
                continue
                
            info = {}
            info["number"] = number
            
            if index < count:
                # 时间
                line = lines[index]
                index += 1
                
                if "-->" in line:
                    timeList = line.split("-->")
                    if len(timeList) == 2:
                        startTime = timeList[0].strip()
                        endTime = timeList[1].strip()
                        info["startTime"] = timecodeToMilliseconds(startTime)
                        info["endTime"] = timecodeToMilliseconds(endTime)
                else:
                    continue

            if index < count:
                # 数据
                line = lines[index]
                index += 1
                
                info["data"] = line.strip()
                char_count = 0
                for ch in line:
                    if ch.isspace():
                        continue
                    if ch in ignore:
                        continue
                    char_count += 1

                info["charCount"] = char_count
                self.totaleCount += char_count

            self.srtInfos.append(info)

        if self.srtInfos:
            lastInfo = self.srtInfos[-1]
            self.totalTime = lastInfo["endTime"]
        
    def updateSrt(self, spaceTime, charTime, minCharCount, scale):
        """更新SRT时间"""
        singleSpaceTime = spaceTime * 1000
        singleCharTime = charTime * 1000

        currentTime = self.srtInfos[0]["startTime"] if self.srtInfos else 0
        for info in self.srtInfos:
            info["startTime"] = currentTime
            
            if info["charCount"] <= minCharCount:
                useTime = info["charCount"] * singleCharTime * scale
            else:
                useTime = info["charCount"] * singleCharTime
                
            if useTime == 0:
                useTime = singleCharTime
                info["endTime"] = info["startTime"] + useTime
                currentTime = info["endTime"] + singleSpaceTime - singleCharTime
            else:
                info["endTime"] = info["startTime"] + useTime
                currentTime = info["endTime"] + singleSpaceTime

        return True
        
    def write(self, srtFile):
        """写入SRT文件"""
        ret = ""
        for info in self.srtInfos:
            ret += str(info["number"]) + "\n"
            ret += millisecondsToTimecode(info["startTime"]) + " --> " + \
                    millisecondsToTimecode(info["endTime"]) + "\n"
            ret += info["data"] + "\n"
            ret += "\n"
        
        with open(srtFile, "w", encoding="utf8") as f:
            f.write(ret)

        return True

    def syncSrtTime(self, refSrt):
        """同步SRT时间"""
        if len(self.srtInfos) != len(refSrt.srtInfos):
            return False

        count = len(self.srtInfos)
        for i in range(count):
            self.srtInfos[i]["startTime"] = refSrt.srtInfos[i]["startTime"]
            self.srtInfos[i]["endTime"] = refSrt.srtInfos[i]["endTime"]

        return True
