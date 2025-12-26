//
//  ToneSandhi.swift
//  MLXAudio
//
//  Mandarin tone sandhi rules for natural pronunciation.
//  Port of PaddleSpeech's tone_sandhi.py
//

import Foundation

/// Mandarin tone sandhi rules for natural pronunciation.
final class ToneSandhi {

    // MARK: - Constants

    private let bu = "不"
    private let yi = "一"

    /// Words that must have neutral tone on last syllable
    private let mustNeuralToneWords: Set<String> = [
        "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰", "馒头", "馄饨", "风筝",
        "难为", "队伍", "阔气", "闺女", "门道", "锄头", "铺盖", "铃铛", "铁匠", "钥匙", "里脊",
        "里头", "部分", "那么", "道士", "造化", "迷糊", "连累", "这么", "这个", "运气", "过去",
        "软和", "转悠", "踏实", "跳蚤", "跟头", "趔趄", "财主", "豆腐", "讲究", "记性", "记号",
        "认识", "规矩", "见识", "裁缝", "补丁", "衣裳", "衣服", "衙门", "街坊", "行李", "行当",
        "蛤蟆", "蘑菇", "薄荷", "葫芦", "葡萄", "萝卜", "荸荠", "苗条", "苗头", "苍蝇", "芝麻",
        "舒服", "舒坦", "舌头", "自在", "膏药", "脾气", "脑袋", "脊梁", "能耐", "胳膊", "胭脂",
        "胡萝", "胡琴", "胡同", "聪明", "耽误", "耽搁", "耷拉", "耳朵", "老爷", "老实", "老婆",
        "戏弄", "将军", "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火", "累赘", "糨糊", "糊涂",
        "精神", "粮食", "簸箕", "篱笆", "算计", "算盘", "答应", "笤帚", "笑语", "笑话", "窟窿",
        "窝囊", "窗户", "稳当", "稀罕", "称呼", "秧歌", "秀气", "秀才", "福气", "祖宗", "砚台",
        "码头", "石榴", "石头", "石匠", "知识", "眼睛", "眯缝", "眨巴", "眉毛", "相声", "盘算",
        "白净", "痢疾", "痛快", "疟疾", "疙瘩", "疏忽", "畜生", "生意", "甘蔗", "琵琶", "琢磨",
        "琉璃", "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务", "牲口", "牙碜", "牌楼", "爽快",
        "爱人", "热闹", "烧饼", "烟筒", "烂糊", "点心", "炊帚", "灯笼", "火候", "漂亮", "滑溜",
        "溜达", "温和", "清楚", "消息", "浪头", "活泼", "比方", "正经", "欺负", "模糊", "槟榔",
        "棺材", "棒槌", "棉花", "核桃", "栅栏", "柴火", "架势", "枕头", "枇杷", "机灵", "本事",
        "木头", "木匠", "朋友", "月饼", "月亮", "暖和", "明白", "时候", "新鲜", "故事", "收拾",
        "收成", "提防", "挖苦", "挑剔", "指甲", "指头", "拾掇", "拳头", "拨弄", "招牌", "招呼",
        "抬举", "护士", "折腾", "扫帚", "打量", "打算", "打扮", "打听", "打发", "扎实", "扁担",
        "戒指", "懒得", "意识", "意思", "悟性", "怪物", "思量", "怎么", "念头", "念叨", "别人",
        "快活", "忙活", "志气", "心思", "得罪", "张罗", "弟兄", "开通", "应酬", "庄稼", "干事",
        "帮手", "帐篷", "希罕", "师父", "师傅", "巴结", "巴掌", "差事", "工夫", "岁数", "屁股",
        "尾巴", "少爷", "小气", "小伙", "将就", "对头", "对付", "寡妇", "家伙", "客气", "实在",
        "官司", "学问", "字号", "嫁妆", "媳妇", "媒人", "婆家", "娘家", "委屈", "姑娘", "姐夫",
        "妯娌", "妥当", "妖精", "奴才", "女婿", "头发", "太阳", "大爷", "大方", "大意", "大夫",
        "多少", "多么", "外甥", "壮实", "地道", "地方", "在乎", "困难", "嘴巴", "嘱咐", "嘟囔",
        "嘀咕", "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴", "哈欠", "哆嗦", "咳嗽", "和尚",
        "告诉", "告示", "含糊", "吓唬", "后头", "名字", "名堂", "合同", "吆喝", "叫唤", "口袋",
        "厚道", "厉害", "千斤", "包袱", "包涵", "匀称", "勤快", "动静", "动弹", "功夫", "力气",
        "前头", "刺猬", "刺激", "别扭", "利落", "利索", "利害", "分析", "出息", "凑合", "凉快",
        "冷战", "冤枉", "冒失", "养活", "关系", "先生", "兄弟", "便宜", "使唤", "佩服", "作坊",
        "体面", "位置", "似的", "伙计", "休息", "什么", "人家", "亲戚", "亲家", "交情", "云彩",
        "事情", "买卖", "主意", "丫头", "丧气", "两口", "东西", "东家", "世故", "不由", "下水",
        "下巴", "上头", "上司", "丈夫", "丈人", "一辈", "那个", "菩萨", "父亲", "母亲", "咕噜",
        "邋遢", "费用", "冤家", "甜头", "介绍", "荒唐", "大人", "泥鳅", "幸福", "熟悉", "计划",
        "扑腾", "蜡烛", "姥爷", "照顾", "喉咙", "吉他", "弄堂", "蚂蚱", "凤凰", "拖沓", "寒碜",
        "糟蹋", "倒腾", "报复", "逻辑", "盘缠", "喽啰", "牢骚", "咖喱", "扫把", "惦记",
    ]

    /// Words that must NOT have neutral tone
    private let mustNotNeuralToneWords: Set<String> = [
        "男子", "女子", "分子", "原子", "量子", "莲子", "石子", "瓜子", "电子", "人人", "虎虎",
        "幺幺", "干嘛", "学子", "哈哈", "数数", "袅袅", "局地", "以下", "娃哈哈", "花花草草", "留得",
        "耕地", "想想", "熙熙", "攘攘", "卵子", "死死", "冉冉", "恳恳", "佼佼", "吵吵", "打打",
        "考考", "整整", "莘莘", "落地", "算子", "家家户户", "青青",
    ]

    /// Punctuation characters
    private let punctuation: Set<Character> = Set("、：，；。？！\u{201C}\u{201D}\u{2018}\u{2019}':,;.?!")

    /// POS tags that should be skipped
    private let skipPosTags: Set<String> = ["x", "eng"]

    // MARK: - Public API

    /// Apply all tone sandhi rules to finals
    func modifiedTone(word: String, pos: String, finals: [String]) -> [String] {
        var result = finals
        result = buSandhi(word: word, finals: result)
        result = yiSandhi(word: word, finals: result)
        result = neuralSandhi(word: word, pos: pos, finals: result)
        result = threeSandhi(word: word, finals: result)
        return result
    }

    /// Pre-merge words for better tone sandhi handling
    func preMergeForModify(segments: [(String, String)]) -> [(String, String)] {
        var result = segments
        result = mergeBu(result)
        result = mergeYi(result)
        result = mergeReduplication(result)
        result = mergeContinuousThreeTones(result)
        result = mergeContinuousThreeTones2(result)
        result = mergeEr(result)
        return result
    }

    // MARK: - Sandhi Rules

    /// 不 tone sandhi: 不 before tone 4 becomes tone 2
    private func buSandhi(word: String, finals: [String]) -> [String] {
        var result = finals
        let chars = Array(word)

        // Pattern: X不X (e.g., 看不懂)
        if chars.count == 3 && chars[1] == Character(bu) {
            result[1] = replaceTone(result[1], with: "5")
        } else {
            for (i, char) in chars.enumerated() {
                if String(char) == bu && i + 1 < chars.count {
                    // 不 before tone 4 becomes tone 2
                    if result[i + 1].last == "4" {
                        result[i] = replaceTone(result[i], with: "2")
                    }
                }
            }
        }

        return result
    }

    /// 一 tone sandhi
    private func yiSandhi(word: String, finals: [String]) -> [String] {
        var result = finals
        let chars = Array(word)

        // Check if it's a number sequence
        if word.contains(yi) && chars.allSatisfy({ String($0) == yi || $0.isNumber }) {
            return result
        }

        // Pattern: X一X (reduplication like 看一看)
        if chars.count == 3 && chars[1] == Character(yi) && chars[0] == chars[2] {
            result[1] = replaceTone(result[1], with: "5")
        }
        // Ordinal: 第一
        else if word.hasPrefix("第一") {
            result[1] = replaceTone(result[1], with: "1")
        } else {
            for (i, char) in chars.enumerated() {
                if String(char) == yi && i + 1 < chars.count {
                    let nextTone = result[i + 1].last
                    if nextTone == "4" || nextTone == "5" {
                        // 一 before tone 4/5 becomes tone 2
                        result[i] = replaceTone(result[i], with: "2")
                    } else if !punctuation.contains(chars[i + 1]) {
                        // 一 before other tones becomes tone 4
                        result[i] = replaceTone(result[i], with: "4")
                    }
                }
            }
        }

        return result
    }

    /// Neural (neutral) tone sandhi
    private func neuralSandhi(word: String, pos: String, finals: [String]) -> [String] {
        guard !mustNotNeuralToneWords.contains(word) else {
            return finals
        }

        var result = finals
        let chars = Array(word)

        // Reduplication words (e.g., 奶奶, 试试)
        for (j, char) in chars.enumerated() {
            if j > 0 && char == chars[j - 1] && ["n", "v", "a"].contains(where: { pos.hasPrefix(String($0)) }) {
                result[j] = replaceTone(result[j], with: "5")
            }
        }

        // 个 as measure word
        if let geIdx = word.firstIndex(of: "个") {
            let idx = word.distance(from: word.startIndex, to: geIdx)
            if idx >= 1 {
                let prevChar = chars[idx - 1]
                if prevChar.isNumber || "几有两半多各整每做是".contains(prevChar) {
                    result[idx] = replaceTone(result[idx], with: "5")
                }
            } else if word == "个" {
                result[0] = replaceTone(result[0], with: "5")
            }
        }

        // Final particles
        if !chars.isEmpty {
            let lastChar = chars[chars.count - 1]
            if "吧呢啊呐噻嘛吖嗨呐哦哒滴哩哟喽啰耶喔诶".contains(lastChar) {
                result[result.count - 1] = replaceTone(result[result.count - 1], with: "5")
            } else if "的地得".contains(lastChar) {
                result[result.count - 1] = replaceTone(result[result.count - 1], with: "5")
            } else if chars.count == 1 && "了着过".contains(lastChar) && ["ul", "uz", "ug"].contains(pos) {
                result[result.count - 1] = replaceTone(result[result.count - 1], with: "5")
            } else if chars.count > 1 && "们子".contains(lastChar) && ["r", "n"].contains(pos) {
                result[result.count - 1] = replaceTone(result[result.count - 1], with: "5")
            } else if chars.count > 1 && "上下".contains(lastChar) && ["s", "l", "f"].contains(pos) {
                result[result.count - 1] = replaceTone(result[result.count - 1], with: "5")
            } else if chars.count > 1 && "来去".contains(lastChar) && chars.count >= 2
                && "上下进出回过起开".contains(chars[chars.count - 2])
            {
                result[result.count - 1] = replaceTone(result[result.count - 1], with: "5")
            }
        }

        // Must neural tone words
        if mustNeuralToneWords.contains(word)
            || (word.count >= 2 && mustNeuralToneWords.contains(String(word.suffix(2))))
        {
            result[result.count - 1] = replaceTone(result[result.count - 1], with: "5")
        }

        return result
    }

    /// Third tone sandhi: consecutive tone 3s
    private func threeSandhi(word: String, finals: [String]) -> [String] {
        var result = finals

        if word.count == 2 && allToneThree(finals) {
            result[0] = replaceTone(result[0], with: "2")
        } else if word.count == 3 && allToneThree(finals) {
            // Determine split point
            let subwords = splitWord(word)
            if subwords[0].count == 2 {
                // disyllabic + monosyllabic
                result[0] = replaceTone(result[0], with: "2")
                result[1] = replaceTone(result[1], with: "2")
            } else {
                // monosyllabic + disyllabic
                result[1] = replaceTone(result[1], with: "2")
            }
        } else if word.count == 4 {
            // Split into two pairs
            let firstPair = Array(result[0..<2])
            let secondPair = Array(result[2..<4])

            if allToneThree(firstPair) {
                result[0] = replaceTone(result[0], with: "2")
            }
            if allToneThree(secondPair) {
                result[2] = replaceTone(result[2], with: "2")
            }
        }

        return result
    }

    // MARK: - Merge Functions

    private func mergeBu(_ segments: [(String, String)]) -> [(String, String)] {
        var result: [(String, String)] = []

        for (i, (word, pos)) in segments.enumerated() {
            if !skipPosTags.contains(pos) {
                if i > 0 && segments[i - 1].0 == bu {
                    let merged = (bu + word, pos)
                    if !result.isEmpty {
                        result.removeLast()
                    }
                    result.append(merged)
                    continue
                }
            }

            let nextPos = i + 1 < segments.count ? segments[i + 1].1 : nil
            if word != bu || nextPos == nil || skipPosTags.contains(nextPos!) {
                result.append((word, pos))
            }
        }

        return result
    }

    private func mergeYi(_ segments: [(String, String)]) -> [(String, String)] {
        var result: [(String, String)] = []
        var skipNext = false

        // Merge reduplication pattern: X一X
        for (i, (word, pos)) in segments.enumerated() {
            if skipNext {
                skipNext = false
                continue
            }

            if i > 0 && word == yi && i + 1 < segments.count && segments[i - 1].0 == segments[i + 1].0
                && segments[i - 1].1 == "v" && !skipPosTags.contains(segments[i + 1].1)
            {
                let merged = result[result.count - 1].0 + yi + segments[i + 1].0
                result[result.count - 1] = (merged, result[result.count - 1].1)
                skipNext = true
            } else {
                result.append((word, pos))
            }
        }

        // Merge 一 with following word
        var finalResult: [(String, String)] = []
        for (word, pos) in result {
            if !finalResult.isEmpty && finalResult[finalResult.count - 1].0 == yi && !skipPosTags.contains(pos) {
                let merged = yi + word
                finalResult[finalResult.count - 1] = (merged, finalResult[finalResult.count - 1].1)
            } else {
                finalResult.append((word, pos))
            }
        }

        return finalResult
    }

    private func mergeReduplication(_ segments: [(String, String)]) -> [(String, String)] {
        var result: [(String, String)] = []

        for (word, pos) in segments {
            if !result.isEmpty && word == result[result.count - 1].0 && !skipPosTags.contains(pos) {
                let merged = result[result.count - 1].0 + word
                result[result.count - 1] = (merged, result[result.count - 1].1)
            } else {
                result.append((word, pos))
            }
        }

        return result
    }

    private func mergeContinuousThreeTones(_ segments: [(String, String)]) -> [(String, String)] {
        // Simplified implementation - merge adjacent all-tone-3 words
        var result: [(String, String)] = []
        var mergeFlags = [Bool](repeating: false, count: segments.count)

        for (i, (word, pos)) in segments.enumerated() {
            if i > 0 && !skipPosTags.contains(pos) && !mergeFlags[i - 1] {
                if word.count + result[result.count - 1].0.count <= 3 && !isReduplication(result[result.count - 1].0) {
                    // Merge
                    let merged = result[result.count - 1].0 + word
                    result[result.count - 1] = (merged, result[result.count - 1].1)
                    mergeFlags[i] = true
                    continue
                }
            }
            result.append((word, pos))
        }

        return result
    }

    private func mergeContinuousThreeTones2(_ segments: [(String, String)]) -> [(String, String)] {
        // Similar to above but checks boundary tones
        return segments  // Simplified - full implementation needs pinyin lookup
    }

    private func mergeEr(_ segments: [(String, String)]) -> [(String, String)] {
        var result: [(String, String)] = []

        for (word, pos) in segments {
            if !result.isEmpty && word == "儿" && !skipPosTags.contains(result[result.count - 1].1) {
                let merged = result[result.count - 1].0 + word
                result[result.count - 1] = (merged, result[result.count - 1].1)
            } else {
                result.append((word, pos))
            }
        }

        return result
    }

    // MARK: - Helpers

    private func replaceTone(_ pinyin: String, with tone: String) -> String {
        guard !pinyin.isEmpty else { return pinyin }
        if let last = pinyin.last, last.isNumber {
            return String(pinyin.dropLast()) + tone
        }
        return pinyin + tone
    }

    private func allToneThree(_ finals: [String]) -> Bool {
        finals.allSatisfy { $0.last == "3" }
    }

    private func splitWord(_ word: String) -> [String] {
        // Simple split - in practice this uses jieba
        let chars = Array(word)
        if chars.count <= 2 {
            return [word]
        }
        // Default: first char separate
        return [String(chars[0]), String(chars[1...])]
    }

    private func isReduplication(_ word: String) -> Bool {
        word.count == 2 && word.first == word.last
    }
}
