"""
字幕对齐核心逻辑 - 从 SW_GenSubTitle/audio_subtitle_gen_and_checker.py 移植
"""
import os
from diff_match_patch import diff_match_patch
import re
from .subtitle_utils import wirite_to_path, word_split_by, replace_symbols_to_one
from .srt_to_fcpxml import SrtsToFcpxml


def process_diffs_with_audio_positions_strong(params):
    """处理差异并计算音频位置"""
    diffs = params['diffs']
    source_text_with_no_info = params['source_text_with_no_info']
    translate_text_dict = params['translate_text_dict']
    generation_subtitle_text = params['generation_subtitle_text']
    source_text_with_info = params['source_text_with_info']
    generation_subtitle_array = params['generation_subtitle_array']
    title = params["title"]
    directory = params["directory"]
    language = params["language"]
    gen_merge_srt = params["gen_merge_srt"]
    source_up_order = params["source_up_order"]
    export_fcpxml = params["export_fcpxml"]
    seamless_fcpxml = params["seamless_fcpxml"]
    source_srt_path = params.get("source_srt_path")
    fcpxml_path = params.get("fcpxml_path")
    
    merge_text = ""
    source_text = ""
    generate_text = ""
    source_store = {}
    generate_store = {}
    allstrings = {}

    def get_source_store(index):
        if index not in source_store:
            source_store[index] = {}
        return source_store[index]

    def get_generate_store(index):
        if index not in generate_store:
            generate_store[index] = {}
        return generate_store[index]

    def get_allstrings(index):
        if index not in allstrings:
            allstrings[index] = {}
        return allstrings[index]

    # 生成对应关系
    for i, (op, content) in enumerate(diffs):
        length_source = len(source_text)
        length_generate = len(generate_text)
        length_all = len(merge_text)
        length_content = len(content)
        
        if op == 1:
            for number in range(0, length_content):
                index_all = number + length_all
                index_source = length_source + number
                source = get_source_store(index_source)
                all = get_allstrings(index_all)
                source["index_in_all"] = index_all
                all["index"] = index_all
                all["source_index"] = index_source
                g = merge_text + content
                all["char"] = source["char"] = g[index_all]
            source_text += content
        elif op == -1:
            for number in range(0, length_content):
                index_all = number + length_all
                index_gen = length_generate + number
                generate = get_generate_store(index_gen)
                all = get_allstrings(index_all)
                all["index"] = generate["index_in_all"] = index_all
                all["gen_index"] = index_gen
                g = merge_text + content
                all["char"] = generate["char"] = g[index_all]
            generate_text += content
        elif op == 0:
            for number in range(0, length_content):
                index_all = number + length_all
                index_source = length_source + number
                source = get_source_store(index_source)
                all = get_allstrings(index_all)
                source["index_in_all"] = index_all
                all["index"] = index_all
                all["source_index"] = index_source
                g = merge_text + content
                all["char"] = source["char"] = g[index_all]
            source_text += content
            
            for number in range(0, length_content):
                index_all = number + length_all
                index_gen = length_generate + number
                generate = get_generate_store(index_gen)
                all = get_allstrings(index_all)
                all["index"] = generate["index_in_all"] = index_all
                all["gen_index"] = index_gen
                g = merge_text + content
                all["char"] = generate["char"] = g[index_all]
            generate_text += content
        merge_text += content

    if len(generate_text) != len(generation_subtitle_text):
        return f"比较生成文件长度不同{len(generate_text)}和{len(generation_subtitle_text)}"
    
    if len(source_text) != len(source_text_with_no_info):
        return f"比较源文件长度不同{len(source_text)}和{len(source_text_with_no_info)}"

    alltextLength = len(merge_text)
    audioEnd = generation_subtitle_array[-1]["audio_end"]
    
    text = ""
    istart = True
    lastpoint = 0
    
    # 从生成文本添加时间戳
    for sentences in generation_subtitle_array:
        for word in sentences["words"]:
            if istart:
                istart = False
                currentContent = word["word"]
                word["whitespace"] = False
            else:
                word["whitespace"] = True
                currentContent = word_split_by["en"] + word["word"]
            
            length_generate = len(text)
            error = None
            if "start" in word:
                start = word["start"]
                end = word["end"]
                score = word["score"]
                wordtext = word["word"]
                whitespace = word["whitespace"]
                eva = (end - start) / len(wordtext) if len(wordtext) > 0 else 0
                if eva > 0.2:
                    error = f"从生成添加时间,长度大{eva}"
                newstart = start
                lastpoint = end
                
                for number in range(0, len(currentContent)):
                    generate = get_generate_store(length_generate + number)
                    index_all = generate["index_in_all"]
                    all = get_allstrings(index_all)
                    all["char_gen_add_in_time"] = currentContent[number]
                    all["wordtext"] = wordtext
                    all["score"] = score
                    all["eva"] = eva
                    
                    if whitespace and number == 0:
                        all["audio_start"] = newstart
                        all["audio_end"] = newstart
                    elif number == len(currentContent) - 1:
                        all["audio_start"] = newstart
                        all["audio_end"] = end
                    else:
                        all["audio_start"] = newstart
                        newstart = round(newstart + eva, 3)
                        all["audio_end"] = newstart
                    if error:
                        all["error"] = error
            else:
                word["error"] = error = "从生成添加时间,没有时间"
                wordtext = word["word"]
                for number in range(0, len(currentContent)):
                    generate = get_generate_store(length_generate + number)
                    index_all = generate["index_in_all"]
                    all = get_allstrings(index_all)
                    all["char_gen_add_in_time"] = currentContent[number]
                    all["wordtext"] = wordtext
                    all["score"] = 0
                    all["eva"] = 0
                    all["audio_start"] = lastpoint
                    all["audio_end"] = lastpoint
                    if error:
                        all["error"] = error

            text += currentContent

    if len(text) != len(generation_subtitle_text):
        return f"从生成添加时间 长度不同{len(text)}和{len(generation_subtitle_text)}"

    merge_text = ""
    lastOp = None
    
    # 源文本添加时间戳  
    for i, (op, content) in enumerate(diffs):
        length_all = len(merge_text)
        length_content = len(content)
        if op == 1:
            if lastOp is None:
                gen_source = None
                audio_durio = 0.1
                if length_all + length_content >= alltextLength:
                    audio_durio = audioEnd
                    gen_source = "结束"
                else:
                    all_end = get_allstrings(length_all + length_content - 1)
                    if "audio_start" in all_end:
                        gen_source = all_end
                        audio_durio = max(all_end["audio_start"] - 0.1, 0)

                eva = audio_durio / length_content if length_content > 0 else 0
                newstart = 0
                for number in range(0, length_content):
                    index_all = number + length_all
                    all = get_allstrings(index_all)
                    all["audio_start"] = newstart
                    newstart = round(newstart + eva, 3)
                    all["audio_end"] = newstart
                    all["eva"] = eva
                    all["from"] = "生成开头丢失文本"
                    all["gen"] = gen_source

            elif lastOp['op'] == 0:
                gen_source = lastOp
                newstart = lastOp["all_end"]["audio_end"]
                audio_end = newstart + 0.2
                if length_all + length_content >= alltextLength:
                    audio_end = audioEnd
                    gen_source = "结束"
                else:
                    all_end = get_allstrings(length_all + length_content - 1)
                    if "audio_start" in all_end:
                        gen_source = all_end
                        audio_end = max(all_end["audio_start"] - 0.1, 0)

                audio_durio = audio_end - newstart
                eva = audio_durio / length_content if length_content > 0 else 0

                for number in range(0, length_content):
                    index_all = number + length_all
                    all = get_allstrings(index_all)
                    all["audio_start"] = newstart
                    newstart = round(newstart + eva, 3)
                    all["audio_end"] = newstart
                    all["eva"] = eva
                    all["from"] = "生成丢失单词"
                    all["gen"] = gen_source

            elif lastOp['op'] == -1:
                all_start = lastOp["all_start"]
                all_end = lastOp["all_end"]
                newstart = all_start["audio_start"]
                audio_durio = all_end["audio_end"] - newstart
                eva = audio_durio / length_content if length_content > 0 else 0
                for number in range(0, length_content):
                    index_all = number + length_all
                    all = get_allstrings(index_all)
                    all["audio_start"] = newstart
                    newstart = round(newstart + eva, 3)
                    all["audio_end"] = newstart
                    all["eva"] = eva
                    all["from"] = "生成错误文本"
                    all["gen"] = lastOp
            lastOp = None
        elif op == -1:
            all_start = get_allstrings(length_all)
            all_end = get_allstrings(length_all + length_content - 1)
            lastOp = {
                "all_start": all_start,
                "all_end": all_end,
                "op": op
            }
        elif op == 0:
            all_start = get_allstrings(length_all)
            all_end = get_allstrings(length_all + length_content - 1)
            lastOp = {
                "all_start": all_start,
                "all_end": all_end,
                "op": op
            }
        merge_text += content

    # 检查所有源文件字符串都赋予时间戳
    for x in allstrings.values():
        if "source_index" in x and "audio_start" not in x:
            return f"检查到没有正常赋值{x}"

    # 每个单词的结尾处添加0.2长度
    sorted_values = [allstrings[key] for key in sorted(allstrings.keys())]
    les = len(sorted_values)
    for i, s in enumerate(sorted_values):
        if 0 < i < les - 1:
            nextitem = sorted_values[i+1]
            if "audio_start" in nextitem and "audio_end" in s:
                current_audio_end = s["audio_end"]
                next_audio_start = nextitem["audio_start"]
                if next_audio_start - current_audio_end >= 0.3:
                    s["audio_end"] = round(s["audio_end"] + 0.1, 3)
                    nextitem["audio_start"] = round(nextitem["audio_start"] - 0.1, 3)

    # 生成字幕文件
    text = ""
    istart = True
    index = 0
    source_srt = ""
    trans_srt = ""
    merge_srt = ""
    
    for content in source_text_with_info["contents"]:
        index += 1
        if istart:
            istart = False
            currentContent = content["content"]
        else:
            currentContent = word_split_by["en"] + content["content"]

        if content["type"] is not None:
            source = get_source_store(len(text))
            endsource = get_source_store(min(len(text + currentContent) - 1, alltextLength - 1))
            if "index_in_all" not in endsource or "index_in_all" not in source:
                print(f"不该发生的错误,找不到存储{content}")
                continue
            start = get_allstrings(source["index_in_all"])
            end = get_allstrings(endsource["index_in_all"])
            if index == 1:
                srt_start = format_time(0)
            else:
                srt_start = format_time(start["audio_start"])
            srt_end = format_time(end["audio_end"])

            merge_transContent = ""
            for k, value in translate_text_dict.items():
                transContentText = value["translate_text_with_info"]["contents"][index-1]['content']
                merge_transContent += transContentText + "\n"
                if "trans_srt" in value:
                    translate_text_dict[k]["trans_srt"] += f"{index}\n{srt_start} --> {srt_end}\n{transContentText}\n\n"
                else:
                    translate_text_dict[k]["trans_srt"] = f"{index}\n{srt_start} --> {srt_end}\n{transContentText}\n\n"
            
            contentText = content['content']
            source_srt += f"{index}\n{srt_start} --> {srt_end}\n{contentText}\n\n"
            
            if gen_merge_srt and merge_transContent != "":
                merge_transContent = merge_transContent.rstrip("\n")
                if source_up_order:
                    merge_srt += f"{index}\n{srt_start} --> {srt_end}\n{contentText}\n{merge_transContent}\n\n"
                else:
                    merge_srt += f"{index}\n{srt_start} --> {srt_end}\n{merge_transContent}\n{contentText}\n\n"

        text += currentContent

    if len(text) != len(source_text_with_no_info):
        return f"核对源文本段落长度不同{len(text)}和{len(source_text_with_no_info)}"

    # 写入原srt（可显式指定完整路径，未指定则保持原命名规则）
    if source_srt_path:
        srt_dir = os.path.dirname(source_srt_path)
        if srt_dir:
            os.makedirs(srt_dir, exist_ok=True)
        with open(source_srt_path, 'w', encoding='utf-8') as f:
            f.write(source_srt)
    else:
        # 默认格式: {文件名}_{语言}_source.srt
        wirite_to_path(source_srt, directory, title + "_" + language + "_source", "srt")

    # 开始写入翻译srt
    for k, value in translate_text_dict.items():
        trans_srt = value["trans_srt"]
        filename = k.replace(".txt", "")
        wirite_to_path(trans_srt, directory, title + "_" + language + "_" + filename + "_translate", "srt")
        
    # 写入合并srt    
    if gen_merge_srt:
        if merge_srt != "":
            wirite_to_path(merge_srt, directory, title + "_" + language + "_merge", "srt")
        else:
            print("翻译文本为空，不再生成合并srt文件。")
            
    if export_fcpxml:
        if fcpxml_path:
            save_path = fcpxml_path
            fcpxml_dir = os.path.dirname(save_path)
            if fcpxml_dir:
                os.makedirs(fcpxml_dir, exist_ok=True)
        else:
            save_path = directory + "/" + title + "_" + language + ".fcpxml"
        
        translate_srt_list = []
        for k, value in translate_text_dict.items():
            trans_srt = value["trans_srt"]
            translate_srt_list.append(trans_srt)
        
        SrtsToFcpxml(source_srt, translate_srt_list, save_path, seamless_fcpxml)
        
    return f"生成了字幕文件{title}"


def format_time(seconds):
    """格式化时间为SRT格式"""
    millis = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def clean_text(text):
    """清理文本"""
    text_no_newlines = re.sub(
        r'\s+', word_split_by["en"], text.strip().replace('\n', '').replace('\r', ''))
    text_cleaned = text_no_newlines.lower()
    text_cleaned = replace_symbols_to_one(text_cleaned)
    return text_cleaned


def audio_subtitle_search_diffent_strong(current_language, directory, file_name, generation_subtitle_array,
                                        generation_subtitle_text, source_text_with_info, translate_text_dict,
                                        gen_merge_srt, source_up_order, export_fcpxml, seamless_fcpxml,
                                        source_srt_path=None, fcpxml_path=None):
    """主对齐函数"""
    source_text_with_no_info = ""
    for content in source_text_with_info["contents"]:
        source_text_with_no_info += word_split_by["en"] + content["content"]
    
    generation_subtitle_text = clean_text(generation_subtitle_text)
    source_text_with_no_info = clean_text(source_text_with_no_info)

    dmp = diff_match_patch()
    dmp.Diff_Timeout = 0
    diffs = dmp.diff_main(generation_subtitle_text, source_text_with_no_info)
    dmp.diff_cleanupSemantic(diffs)

    return process_diffs_with_audio_positions_strong({
        "title": file_name,
        'diffs': diffs,
        "directory": directory,
        "language": current_language,
        "gen_merge_srt": gen_merge_srt,
        "source_up_order": source_up_order,
        "export_fcpxml": export_fcpxml,
        "seamless_fcpxml": seamless_fcpxml,
        "source_srt_path": source_srt_path,
        "fcpxml_path": fcpxml_path,
        'source_text_with_no_info': source_text_with_no_info,
        "translate_text_dict": translate_text_dict,
        'generation_subtitle_text': generation_subtitle_text,
        'source_text_with_info': source_text_with_info,
        'generation_subtitle_array': generation_subtitle_array
    })
