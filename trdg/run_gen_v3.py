from PIL import Image
import bitstring
import jaconv
import pandas as pd

import numpy as np
from data_aug.data_aug import *
from data_aug.bbox_util import *
 
import cv2 
import matplotlib.pyplot as plt 
import pickle as pkl

import cv2
import math
import os
import random as rnd

import skimage.filters as filters
import skimage
from straug.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from straug.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
from straug.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color

from tqdm import tqdm

DATA_DIR_ROOT = '/home/anlab/Tienanh-backup/TrainHWJapanese/data/ETLCDB/'
NUMBER_DATASET = 160

data_format = None

class ETLn_Record:
    def read(self, bs, pos=None):
        if pos:
            bs.bytepos = pos * self.octets_per_record

        r = bs.readlist(self.bitstring)

        record = dict(zip(self.fields, r))

        self.record = {
            k: (self.converter[k](v) if k in self.converter else v)
            for k, v in record.items()
        }

        return self.record

    def get_image(self):
        return self.record['Image Data']

class ETL167_Record(ETLn_Record):
    def __init__(self):
        self.octets_per_record = 2052
        self.fields = [
            "Data Number", "Character Code", "Serial Sheet Number", "JIS Code", "EBCDIC Code",
            "Evaluation of Individual Character Image", "Evaluation of Character Group",
            "Male-Female Code", "Age of Writer", "Serial Data Number",
            "Industry Classification Code", "Occupation Classification Code",
            "Sheet Gatherring Date", "Scanning Date",
            "Sample Position Y on Sheet", "Sample Position X on Sheet",
            "Minimum Scanned Level", "Maximum Scanned Level", "Image Data"
        ]
        self.bitstring = 'uint:16,bytes:2,uint:16,hex:8,hex:8,4*uint:8,uint:32,4*uint:16,4*uint:8,pad:32,bytes:2016,pad:32'
        self.converter = {
            'Character Code': lambda x: x.decode('ascii'),
            'Image Data': lambda x: Image.eval(Image.frombytes('F', (64, 63), x, 'bit', 4).convert('L'),
                                               lambda x: x * 16)
        }

    def get_char(self):
        return bytes.fromhex(self.record['JIS Code']).decode('shift_jis')

class ETL8G_Record(ETLn_Record):
    def __init__(self):
        self.octets_per_record = 8199
        self.fields = [
            "Serial Sheet Number", "JIS Kanji Code", "JIS Typical Reading", "Serial Data Number",
            "Evaluation of Individual Character Image", "Evaluation of Character Group",
            "Male-Female Code", "Age of Writer",
            "Industry Classification Code", "Occupation Classification Code",
            "Sheet Gatherring Date", "Scanning Date",
            "Sample Position X on Sheet", "Sample Position Y on Sheet", "Image Data"
        ]
        self.bitstring = 'uint:16,hex:16,bytes:8,uint:32,4*uint:8,4*uint:16,2*uint:8,pad:240,bytes:8128,pad:88'
        self.converter = {
            'JIS Typical Reading': lambda x: x.decode('ascii'),
            'Image Data': lambda x: Image.eval(Image.frombytes('F', (128, 127), x, 'bit', 4).convert('L'),
            lambda x: x * 16)
        }
    
    def get_char(self):
        char = bytes.fromhex(
            '1b2442' + self.record['JIS Kanji Code'] + '1b2842').decode('iso2022_jp')
        return char

class ETL9G_Record(ETLn_Record):
    def __init__(self):
        self.octets_per_record = 8199
        self.fields = [
            "Serial Sheet Number", "JIS Kanji Code", "JIS Typical Reading", "Serial Data Number",
            "Evaluation of Individual Character Image", "Evaluation of Character Group",
            "Male-Female Code", "Age of Writer",
            "Industry Classification Code", "Occupation Classification Code",
            "Sheet Gatherring Date", "Scanning Date",
            "Sample Position X on Sheet", "Sample Position Y on Sheet", "Image Data"
        ]
        self.bitstring = 'uint:16,hex:16,bytes:8,uint:32,4*uint:8,4*uint:16,2*uint:8,pad:272,bytes:8128,pad:56'
        self.converter = {
            'JIS Typical Reading': lambda x: x.decode('ascii'),
            'Image Data': lambda x: Image.eval(Image.frombytes('F', (128, 127), x, 'bit', 4).convert('L'),
                                               lambda x: x * 16)
        }

    def get_char(self):
        char = bytes.fromhex(
            '1b2442' + self.record['JIS Kanji Code'] + '1b2842').decode('iso2022_jp')
        return char

def read_chars(filename):
    text_file= open(filename,'r')
    data=text_file.read()
    return "".join(data.split())

char_dict_8G = {}
char_dict_9G = {}
char_dict_1C = {}



def white_bg_square(img):
    "return a white-background-color image having the img in exact center"
    size = (max(img.size),)*2
    layer = Image.new('RGB', size, (255,255,255))
    layer.paste(img, tuple(map(lambda x:(x[0]-x[1])/2, zip(size, img.size))))
    return layer

def del_dup(s):
   result = ""
   
   for char in s:
       
       if char not in result:
           result += char
   return result


df = pd.read_csv('/home/anlab/Tienanh-backup/TrainHWJapanese/data/address_lv2.csv', header=None, encoding='utf-8')
df[0]
list = df[0].to_list()
listToStr = ''.join([str(elem) for elem in list])
listToStr = listToStr + '北海道青森県岩手県宮城県秋田県山形県福島県茨城県栃木県群馬県埼玉県千葉県東京都神奈川県新潟県富山県石川県福井県山梨県長野県岐阜県静岡県愛知県三重県滋賀県京都府大阪府兵庫県奈良県和歌山県鳥取県島根県岡山県広島県山口県徳島県香川県愛媛県高知県福岡県佐賀県長崎県熊本県大分県宮崎県鹿児島県沖縄県'
new_str= del_dup(listToStr)

BIG_CHAR='あ愛委壱雲円王何火会階革官館希い記休牛共曲句係芸券険古語口港う合査菜刷蚕始死資治質弱需習出え暑小章状深図政製責舌選倉増属お多隊短竹貯直提転土等童内熱馬か畑番美評布福平弁法毎名目輸容が浴略緑列話きゃ悪意印営園黄価花解貝学幹岸ぎ旗貴宮去協極区兄欠建験固誤向く皇告再際察賛姉氏歯示実主収衆ぐ術書少証植申推整西赤先銭想蔵け族太代団茶丁賃程点党答道南年げ拝八否鼻病府複陛保豊末命問勇こ曜来流林練ご圧易員栄延億加荷回外楽感眼期さ起急居境玉苦型決憲元己護后耕ざ国最在殺酸子私事耳舎取周週述しゅ諸承象織真水星誠切千前早造じ続打台断着帳追敵伝冬統銅難念す敗発悲必秒父仏米歩暴万明門友ず様落留臨連せ安異因永演屋可課快害額慣岩機ぜ技救挙強勤具形潔検原庫交孝考そ穀妻材雑残市糸似自写守宗集春ぞ助招賞職神数晴青接宣善争側卒た体大男中張通的田刀討得二燃配だ判比筆品負物別補望満盟夜有ちょ洋利旅輪路ぢ暗移引泳遠恩夏貨改各活歓顔帰つ疑求許教均空敬結権厳戸候工航づ黒才罪三仕師紙児辞社手就住準て女昭上色臣世正静折専然相則存で対第談忠朝低適電島頭徳弐納倍と半皮百貧武分変墓貿味迷野由用ど理両類労な案胃飲英塩温家過械拡株漢願気にっ義球漁橋禁君景血犬減故光幸ぬ行今採財参使志至字式者種州十ね純序消乗食親是清税設川全総息の孫帯題知昼町停鉄徒投働特肉能は買反肥俵不部奮編母防未鳴役遊ば葉里料令老ぱ以遺院衛央音科我海格寒管喜汽ひ議究魚興近訓系月研現湖公広講び根済坂山司思視寺識謝酒修従順ぴ除焼場信身制生席節戦祖草測尊ふ待達地柱腸定典登東動毒日農売ぶ板費標付風粉辺包北脈綿約夕要ぷ陸良例六へ位医右液往下果画界確刊観器季べ客級京鏡金群経件絹言五功康鉱ぺ混災作散史指詞持七車首拾重処ほ勝照常心進勢精石説浅素走足損ぼ態谷池注調底天都湯同独入波博ぽ版非氷夫復文返報牧民面薬予陽ま律量冷録み依育雨益応化歌芽絵覚勧間基紀む逆給供業銀軍計健県限午効校高め左祭昨産四支詩時失借受秋宿初も商省情新人性聖積雪線組送速村や貸単置虫長庭展努燈堂読任派白ゆ犯飛票婦幅聞便放本務毛訳余養よ率領礼論ら囲一運駅横仮河賀開角完関寄規り久旧競局九郡軽兼見個後厚構号る差細策算士止試次室釈授終祝所れ唱称条森仁成声績絶船創像俗他ろ退炭築著鳥弟店度当導届認破麦わ飯備表富服兵勉方妹無木油預欲を立力歴和ん亜唖娃阿哀愛挨姶逢葵茜穐悪握渥あ旭葦芦鯵梓圧斡扱宛姐虻飴絢綾鮎い或粟袷安庵按暗案闇鞍杏以伊位依う偉囲夷委威尉惟意慰易椅為畏異移維緯胃萎衣謂違遺医井亥域育郁磯一壱溢逸稲茨芋鰯允印咽員因姻引飲淫胤蔭院陰隠韻吋右宇烏羽迂雨卯鵜窺丑碓臼渦嘘唄欝蔚鰻姥厩浦瓜閏噂云運雲荏餌叡営嬰影映曳栄永泳洩瑛盈穎頴英衛詠鋭液疫益駅悦謁越閲榎厭円園堰奄宴延怨掩え援沿演炎焔煙燕猿縁艶苑薗遠鉛鴛お塩於汚甥凹央奥往応押旺横欧殴王か翁襖鴬鴎黄岡沖荻億屋憶臆桶牡乙俺卸恩温穏音下化仮何伽価佳加可嘉夏嫁家寡科暇果架歌河火珂禍禾稼箇花苛茄荷華菓蝦課嘩貨迦過霞蚊俄峨我牙画臥芽蛾賀雅餓駕介会解回塊壊廻快怪悔恢懐戒拐改魁晦械海灰界皆絵芥蟹開階貝凱劾外咳害崖慨概涯碍蓋街該鎧骸浬馨蛙が垣柿蛎鈎劃嚇各廓拡撹格核殻獲確き穫覚角赫較郭閣隔革学岳楽額顎掛ぎ笠樫橿梶鰍潟割喝恰括活渇滑葛褐轄且鰹叶椛樺鞄株兜竃蒲釜鎌噛鴨栢茅萱粥刈苅瓦乾侃冠寒刊勘勧巻喚堪姦完官寛干幹患感慣憾換敢柑桓棺款歓汗漢澗潅環甘監看竿管簡緩缶翰肝艦莞観諌貫還鑑間閑関陥韓館舘丸含岸巌玩癌眼岩翫贋雁頑顔願企伎危喜器基奇嬉寄岐希幾く忌揮机旗既期棋棄機帰毅気汽畿祈ぐ季稀紀徽規記貴起軌輝飢騎鬼亀偽け儀妓宜戯技擬欺犠疑祇義蟻誼議掬菊鞠吉吃喫桔橘詰砧杵黍却客脚虐逆丘久仇休及吸宮弓急救朽求汲泣灸球究窮笈級糾給旧牛去居巨拒拠挙渠虚許距鋸漁禦魚亨享京供侠僑兇競共凶協匡卿叫喬境峡強彊怯恐恭挟教橋況狂狭矯胸脅興蕎郷鏡響饗驚仰凝尭暁業局曲極玉桐粁僅げ勤均巾錦斤欣欽琴禁禽筋緊芹菌衿こ襟謹近金吟銀九倶句区狗玖矩苦躯ご駆駈駒具愚虞喰空偶寓遇隅串櫛釧屑屈掘窟沓靴轡窪熊隈粂栗繰桑鍬勲君薫訓群軍郡卦袈祁係傾刑兄啓圭珪型契形径恵慶慧憩掲携敬景桂渓畦稽系経継繋罫茎荊蛍計詣警軽頚鶏芸迎鯨劇戟撃激隙桁傑欠決潔穴結血訣月件倹倦健兼券剣喧圏堅嫌建憲懸拳捲検権牽犬献研硯絹さ県肩見謙賢軒遣鍵険顕験鹸元原厳ざ幻弦減源玄現絃舷言諺限乎個古呼し固姑孤己庫弧戸故枯湖狐糊袴股胡じ菰虎誇跨鈷雇顧鼓五互伍午呉吾娯後御悟梧檎瑚碁語誤護醐乞鯉交佼侯候倖光公功効勾厚口向后喉坑垢好孔孝宏工巧巷幸広庚康弘恒慌抗拘控攻昂晃更杭校梗構江洪浩港溝甲皇硬稿糠紅紘絞綱耕考肯肱腔膏航荒行衡講貢購郊酵鉱砿鋼閤す降項香高鴻剛劫号合壕拷濠豪轟麹ず克刻告国穀酷鵠黒獄漉腰甑忽惚骨せ狛込此頃今困坤墾婚恨懇昏昆根梱ぜ混痕紺艮魂些佐叉唆嵯左差査沙瑳砂詐鎖裟坐座挫債催再最哉塞妻宰彩才採栽歳済災采犀砕砦祭斎細菜裁載際剤在材罪財冴坂阪堺榊肴咲崎埼碕鷺作削咋搾昨朔柵窄策索錯桜鮭笹匙冊刷察拶撮擦札殺薩雑皐鯖捌錆鮫皿晒三傘参山惨撒散そ桟燦珊産算纂蚕讃賛酸餐斬暫残仕ぞ仔伺使刺司史嗣四士始姉姿子屍市た師志思指支孜斯施旨枝止死氏獅祉だ私糸紙紫肢脂至視詞詩試誌諮資賜雌飼歯事似侍児字寺慈持時次滋治爾璽痔磁示而耳自蒔辞汐鹿式識鴫竺軸宍雫七叱執失嫉室悉湿漆疾質実蔀篠偲柴芝屡蕊縞舎写射捨赦斜煮社紗者謝車遮蛇邪借勺尺杓灼爵酌釈錫若寂弱惹主取守手朱殊ち狩珠種腫趣酒首儒受呪寿授樹綬需ぢ囚収周宗就州修愁拾洲秀秋終繍習つ臭舟蒐衆襲讐蹴輯週酋酬集醜什住づ充十従戎柔汁渋獣縦重銃叔夙宿淑祝縮粛塾熟出術述俊峻春瞬竣舜駿准循旬楯殉淳準潤盾純巡遵醇順処初所暑曙渚庶緒署書薯藷諸助叙女序徐恕鋤除傷償勝匠升召哨商唱嘗奨妾娼宵将小少尚庄床廠彰承抄招掌捷昇昌昭晶松梢樟樵沼消渉て湘焼焦照症省硝礁祥称章笑粧紹肖で菖蒋蕉衝裳訟証詔詳象賞醤鉦鍾鐘と障鞘上丈丞乗冗剰城場壌嬢常情擾ど条杖浄状畳穣蒸譲醸錠嘱埴飾拭植殖燭織職色触食蝕辱尻伸信侵唇娠寝審心慎振新晋森榛浸深申疹真神秦紳臣芯薪親診身辛進針震人仁刃塵壬尋甚尽腎訊迅陣靭笥諏須酢図厨逗吹垂帥推水炊睡粋翠衰遂酔錐錘随瑞髄崇嵩数枢趨雛据杉椙な菅頗雀裾澄摺寸世瀬畝是凄制勢姓に征性成政整星晴棲栖正清牲生盛精ぬ聖声製西誠誓請逝醒青静斉税脆隻ね席惜戚斥昔析石積籍績脊責赤跡蹟碩切拙接摂折設窃節説雪絶舌蝉仙先千占宣専尖川戦扇撰栓栴泉浅洗染潜煎煽旋穿箭線繊羨腺舛船薦詮賎践選遷銭銑閃鮮前善漸然全禅繕膳糎噌塑岨措曾曽楚狙疏疎礎祖租粗素組蘇訴阻遡鼠僧創双叢倉の喪壮奏爽宋層匝惣想捜掃挿掻操早は曹巣槍槽漕燥争痩相窓糟総綜聡草ば荘葬蒼藻装走送遭鎗霜騒像増憎臓ぱ蔵贈造促側則即息捉束測足速俗属賊族続卒袖其揃存孫尊損村遜他多太汰詑唾堕妥惰打柁舵楕陀駄騨体堆対耐岱帯待怠態戴替泰滞胎腿苔袋貸退逮隊黛鯛代台大第醍題鷹滝瀧卓啄宅托択拓沢濯琢託鐸濁諾茸凧蛸只叩但達辰奪脱巽竪辿棚ひ谷狸鱈樽誰丹単嘆坦担探旦歎淡湛び炭短端箪綻耽胆蛋誕鍛団壇弾断暖ぴ檀段男談値知地弛恥智池痴稚置致ふ蜘遅馳築畜竹筑蓄逐秩窒茶嫡着中仲宙忠抽昼柱注虫衷註酎鋳駐樗瀦猪苧著貯丁兆凋喋寵帖帳庁弔張彫徴懲挑暢朝潮牒町眺聴脹腸蝶調諜超跳銚長頂鳥勅捗直朕沈珍賃鎮陳津墜椎槌追鎚痛通塚栂掴槻佃漬柘辻蔦綴鍔椿潰坪壷嬬紬爪吊釣ぶ鶴亭低停偵剃貞呈堤定帝底庭廷弟ぷ悌抵挺提梯汀碇禎程締艇訂諦蹄逓へ邸鄭釘鼎泥摘擢敵滴的笛適鏑溺哲べ徹撤轍迭鉄典填天展店添纏甜貼転顛点伝殿澱田電兎吐堵塗妬屠徒斗杜渡登菟賭途都鍍砥砺努度土奴怒倒党冬凍刀唐塔塘套宕島嶋悼投搭東桃梼棟盗淘湯涛灯燈当痘祷等答筒糖統到董蕩藤討謄豆踏逃透鐙陶頭騰闘働動同堂導憧撞洞瞳童ぺ胴萄道銅峠鴇匿得徳涜特督禿篤毒ほ独読栃橡凸突椴届鳶苫寅酉瀞噸屯ぼ惇敦沌豚遁頓呑曇鈍奈那内乍凪薙ぽ謎灘捺鍋楢馴縄畷南楠軟難汝二尼弐迩匂賑肉虹廿日乳入如尿韮任妊忍認濡禰祢寧葱猫熱年念捻撚燃粘乃廼之埜嚢悩濃納能脳膿農覗蚤巴把播覇杷波派琶破婆罵芭馬俳廃拝排敗杯盃牌背肺輩配倍培媒梅楳煤狽買売賠陪這蝿秤矧萩伯剥博ま拍柏泊白箔粕舶薄迫曝漠爆縛莫駁み麦函箱硲箸肇筈櫨幡肌畑畠八鉢溌む発醗髪伐罰抜筏閥鳩噺塙蛤隼伴判め半反叛帆搬斑板氾汎版犯班畔繁般藩販範釆煩頒飯挽晩番盤磐蕃蛮匪卑否妃庇彼悲扉批披斐比泌疲皮碑秘緋罷肥被誹費避非飛樋簸備尾微枇毘琵眉美鼻柊稗匹疋髭彦膝菱肘弼必畢筆逼桧姫媛紐百謬俵彪標氷漂瓢票表評豹廟描病秒苗錨鋲も蒜蛭鰭品彬斌浜瀕貧賓頻敏瓶不付や埠夫婦富冨布府怖扶敷斧普浮父符ゆ腐膚芙譜負賦赴阜附侮撫武舞葡蕪部封楓風葺蕗伏副復幅服福腹複覆淵弗払沸仏物鮒分吻噴墳憤扮焚奮粉糞紛雰文聞丙併兵塀幣平弊柄並蔽閉陛米頁僻壁癖碧別瞥蔑箆偏変片篇編辺返遍便勉娩弁鞭保舗鋪圃捕歩甫補輔穂募墓慕戊暮母簿菩倣俸包呆報奉宝峰峯崩庖抱捧放よ方朋法泡烹砲縫胞芳萌蓬蜂褒訪豊ら邦鋒飽鳳鵬乏亡傍剖坊妨帽忘忙房り暴望某棒冒紡肪膨謀貌貿鉾防吠頬北僕卜墨撲朴牧睦穆釦勃没殆堀幌奔本翻凡盆摩磨魔麻埋妹昧枚毎哩槙幕膜枕鮪柾鱒桝亦俣又抹末沫迄侭繭麿万慢満漫蔓味未魅巳箕岬密蜜湊蓑稔脈妙粍民眠務夢無牟矛霧鵡椋婿娘冥名命明盟迷銘鳴姪牝滅免棉綿緬面麺摸模茂妄孟毛猛る盲網耗蒙儲木黙目杢勿餅尤戻籾貰れ問悶紋門匁也冶夜爺耶野弥矢厄役ろ約薬訳躍靖柳薮鑓愉愈油癒諭輸唯佑優勇友宥幽悠憂揖有柚湧涌猶猷由祐裕誘遊邑郵雄融夕予余与誉輿預傭幼妖容庸揚揺擁曜楊様洋溶熔用窯羊耀葉蓉要謡踊遥陽養慾抑欲沃浴翌翼淀羅螺裸来莱頼雷洛絡落酪乱卵嵐欄濫藍蘭覧利吏履李梨理璃痢裏裡里離陸律率立葎掠略わ劉流溜琉留硫粒隆竜龍侶慮旅虜了を亮僚両凌寮料梁涼猟療瞭稜糧良諒ん遼量陵領力緑倫厘林淋燐琳臨輪隣鱗麟瑠塁涙累類令伶例冷励嶺怜玲礼苓鈴隷零霊麗齢暦歴列劣烈裂廉恋憐漣煉簾練聯蓮連錬呂魯櫓炉賂路露労婁廊弄朗楼榔浪漏牢狼篭老聾蝋郎六麓禄肋録論倭和話歪賄脇惑枠鷲亙亘鰐詫藁蕨椀湾碗腕'
new_str = sorted(new_str)
BIG_CHAR = sorted(BIG_CHAR)
have_same = ''
dont_have = ''
for char in new_str:
    if char in BIG_CHAR:
        have_same = have_same + char
    else:
        dont_have = dont_have + char    

label_ungen = []
for string in list:
    for char in string:
        if char in dont_have:
            label_ungen.append(string)
            break

labels = []
for string in list:
    if string not in label_ungen:
        labels.append(string)

img_4 = np.zeros([127,20,3],dtype=np.uint8)
img_4.fill(255) 
white_blank_space_lv4 = Image.fromarray(img_4)



bboxes = pkl.load(open("/home/anlab/Tienanh-backup/License_plate/messi_ann.pkl", "rb"))


def resize(image_pil, width, height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')


def white_space_random(images):
    new_images = []
    for l in range(len(images)):
        img = Image.eval(images[l], lambda x: 255 - x)
        img = img.crop((28,0,128,127))

        distance = np.random.randint(-30,0)
        
        new_width = img.width + distance
        new_img = Image.new("RGB", (new_width, 127), "white")
        new_img.paste(img,(0,0))
        new_images.append(new_img)
    image_rotate = []
    for img in new_images:
        # img = np.array(img)
        # seq = Sequence([RandomRotate(8)])
        # img, _ = seq(img, bboxes)
        # print(img.size)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        choice_all = random.randint(0,1)

        if choice_all == 0:
            choice = random.uniform(0,1)

            if choice >= 0.8 and choice<=1:
                kernel = np.ones((2,2), np.uint8)
                img = cv2.erode(img, kernel, iterations=2)
            else: 
                kernel = np.ones((4,4), np.uint8)
                img = cv2.erode(img, kernel, iterations=1)

        elif choice_all == 1: 

            kernel = np.ones((3,3), np.uint8)
            img = cv2.dilate(img, kernel, iterations=1)

            kernel = np.ones((3,3), np.uint8)
            img = cv2.erode(img, kernel, iterations=1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        choice = 0.65
        img = resize(img, int(img.width*choice), int(img.height*choice))
        
        white = (255,255,255)
        angle = random.uniform(-10,10)
        img = img.rotate(angle, 0, expand = 1, fillcolor = white)
        # print(img.size)
        
        # img.show()
        # img = Image.fromarray(img)
        # img.show()
        image_rotate.append(img)    
        
    return image_rotate
def produce_image_bg(height, width, image_dir):
    """
        Create a background with a image
    """
    images = os.listdir(image_dir)

    if len(images) > 0:
        pic = Image.open(
            os.path.join(image_dir, images[rnd.randint(0, len(images) - 1)])
        )

        if pic.size[0] < width:
            pic = pic.resize(
                [width, int(pic.size[1] * (width / pic.size[0]))], Image.ANTIALIAS
            )
        if pic.size[1] < height:
            pic = pic.resize(
                [int(pic.size[0] * (height / pic.size[1])), height], Image.ANTIALIAS
            )

        if pic.size[0] == width:
            x = 0
        else:
            x = rnd.randint(0, pic.size[0] - width)
        if pic.size[1] == height:
            y = 0
        else:
            y = rnd.randint(0, pic.size[1] - height)

        return pic.crop((x, y, x + width, y + height))
    else:
        raise Exception("No images where found in the images folder!")




# Cấu hình 2 08062022 - fixed
# Cấu hình 2
# Cấu hình 2
def generate(text, count, check):
    if len(char_dict_8G) == 0:
        chars = read_chars(DATA_DIR_ROOT + 'ETL8G/chars.txt')
        # print("number char 8G = ", len(chars))
        for i in range(len(chars)):
            char_dict_8G[chars[i]] = i
    if len(char_dict_9G) == 0:
        chars = read_chars(DATA_DIR_ROOT + 'ETL9G/chars.txt')
        # print("number char 9G = ", len(chars))
        for i in range(len(chars)):
            char_dict_9G[chars[i]] = i
    
    
    images=[]
    for k in range(len(text)):
        if text[k] in char_dict_8G:
            indexDataFile = count // 5 + 1
            dataFile = DATA_DIR_ROOT + 'ETL8G/ETL8G_{:02d}'.format(indexDataFile)
            # print("dataFile = ", dataFile)
            etln_record = ETL8G_Record()
            index = char_dict_8G[text[k]] + 956*(count%5)
            # print("index=", index)
            f = bitstring.ConstBitStream(filename=dataFile)
            record = etln_record.read(f, index)
            char = etln_record.get_char()
            img = etln_record.get_image()
            images.append(img)
        
        elif text[k] in char_dict_9G:
            indexDataFile = count // 4 + 1
            dataFile = DATA_DIR_ROOT + 'ETL9G/ETL9G_{:02d}'.format(indexDataFile)
            # print("dataFile = ", dataFile)
            etln_record = ETL9G_Record()
            index = char_dict_9G[text[k]] + 3036*(count%4)
            # print("index=", index)
            f = bitstring.ConstBitStream(filename=dataFile)
            record = etln_record.read(f, index)
            char = etln_record.get_char()
            img = etln_record.get_image()
            images.append(img)

        elif text[k] in char_dict_1C:
            etln_record = ETL167_Record()
        elif text[k]=='_':
            continue
        else:
            images.clear()
            break
    if len(images) > 0:
        images = white_space_random(images)
        img_transform = []
        boxes = []
        for new_img in images:
            new_img = np.array(new_img)
            # print(new_img.shape)
            # cv2.imwrite('or.jpg',new_img )
            new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
            # print(new_img.shape)
            m = cv2.mean(new_img)
            thresh = min(200, m[0])
            _, new_img = cv2.threshold(new_img, thresh, 255, cv2.THRESH_BINARY_INV)
            # cv2.imwrite('new.jpg',new_img)
            coords = cv2.findNonZero(new_img)
            x_coord,y_coord,w_coord,h_coord = cv2.boundingRect(coords)
            boxes.append((x_coord,y_coord,w_coord,h_coord))


        w = 0
        h_ = []
        char_images = images
        for i in range(len(char_images)):
            x_coord, y_coord, w_coord, h_coord = boxes[i]
            # print(x_coord, y_coord, w_coord, h_coord)
            
            new_img = np.array(char_images[i])
             

            if x_coord == 0 or y_coord == 0 or w_coord == 0 or h_coord == 0:
                crop_image = new_img
                crop_image = Image.fromarray(crop_image)
            else:
                crop_image = new_img[y_coord:y_coord+h_coord, x_coord:x_coord+w_coord]
                crop_image = Image.fromarray(crop_image)

            w = w + crop_image.width 
            h_.append(crop_image.height)
            
            img_transform.append(crop_image)    
        max_h = max(h_)
        # print(w)
        # w, h = images[0].width, images[0].height
        # index = 0
        # for i in range(len(text)):
        #     if text.isspace():
        #         w = w + white_blank_space_lv4.width
        #         index = i
        #     else:
        #         continue
        
        
        tiled = Image.new("RGB", (w, max_h), "white")
        tiled.save("bg.png")

        # tiled = produce_image_bg(max_h , w, '/home/anlab/Tienanh-backup/TrainHWJapanese/TextRecognitionDataGenerator/trdg/bg_img')
        # tiled.save("bg.png")
        
        img_result = []
        h_after = []
        w_result = 0
        boxes_after = []
        for l in range(len(img_transform)):
            img = Image.eval(img_transform[l], lambda x: x)
            start_point = (0, 0)
            end_point = (img.width,img.height)
            # img_array = np.array(img)
            # img_array = cv2.rectangle(img_array, start_point, end_point, (255,0,0), 1)
            # cv2.imwrite("boudingbox.jpg", img_array)
            if check:
                new_crop_img = Image.new("RGB", (img.width+15, img.height+10), "white")
                rand_dis_1 = random.randint(0,15)
                rand_dis_2 = random.randint(0,10)
                new_crop_img.paste(img,(rand_dis_1,rand_dis_2))
            else:
                new_crop_img = Image.new("RGB", (img.width+10, img.height), "white")
                rand_dis_1 = random.randint(0,10)
                rand_dis_2 = 0
                new_crop_img.paste(img,(rand_dis_1,rand_dis_2))

            img_after = np.array(new_crop_img)
            # img_after = cv2.rectangle(img_after, (rand_dis_1, rand_dis_2), ((rand_dis_1+img.width),(rand_dis_2+img.height)), (255,0,0), 1)
            boxes_after.append((rand_dis_1, rand_dis_2, new_crop_img.width, img.height))
            # print(boxes_after)
            cv2.imwrite('fucking_stupid_object.jpg', img_after)
            img_after = Image.fromarray(img_after)
            
            # tiled.paste(img, (next_w, 0))
            # next_w = next_w + img_transform[l].width + 15


            img_result.append(img_after)
            h_after.append(img_after.height)
            w_result = w_result + img_after.width
        height_result = max(h_after)
        

        tiled = Image.new("RGB", (w_result, height_result), "white")
        next_w = 0
        x0, y0, w0, h0 = boxes_after[0]
        x1, y1, w1, h1 = boxes_after[1]
        x2, y2, w2, h2 = boxes_after[2]
        for u in range(len(img_result)):

            img = Image.eval(img_result[u], lambda x: x)
            tiled.paste(img, (next_w, 0))
            next_w = next_w + img_result[u].width
        
        x_top = w0 + w1 + x2
        y_top = y2
        real_w = w2 -20
        # print(x_top, y_top)
        image_array = np.array(tiled)
        image_array = cv2.rectangle(image_array, (x_top, y_top), ((x_top+real_w),(y_top+h2)), (255,0,0), 2)
        # tiled = Image.fromarray(image_array)
        cv2.imwrite('image_final.jpg', image_array)
        tiled.save("tiledfn.png")
        #create RGBA image and RGB mask
        image = tiled.convert("RGBA") #Image.new("RGBA", (tiled.width, tiled.height), (0,0,0,0))
        datas = image.getdata()
        newData = []
        for item in datas:
            if item[0] >= 250 and item[1] >= 250 and item[2] >= 250:
                newData.append((item[0], item[1], item[2], 0))
            else:
                newData.append(item)
        image.putdata(newData)
        mask = tiled.convert("RGB")#Image.new("RGB", (tiled.width, tiled.height), (0, 0, 0))

    else:
        print("cannot generate this text: ", text)
        image = Image.new("RGBA", (50, 50), (0,0,0,0))
        mask = Image.new("RGB", (50, 50), (0, 0, 0))

    return mask, x_top, y_top, x_top+real_w, y_top+h2, mask.width, mask.height


def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return (((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h)

count = 0
# l = '福山市', '犬山市', '大阪市東住吉区', '高萩市', '大田区', '岡山市中区', '神戸市灘区', '上川郡鷹栖町', '一宮市', '坂東市'
# labels = ['愛知県一宮市', '宮城県仙台市泉区','茨城県古河市', 
# '愛知県名古屋市中川区', '広島県呉市', '福岡県嘉麻市', '茨城県坂東市', '滋賀県大津市', 
# '滋賀県守山市', '山形県寒河江市', '岐阜県岐阜市', '岡山県岡山市中区', '千葉県市川市', '青森県平川市', '徳島県徳島市', '新潟県新潟市江南区',
# '三重県松阪市', '愛知県犬山市', '兵庫県神戸市灘区', '広島県福山市', '兵庫県西宮市', '滋賀県近江八幡市',
# '茨城県高萩市']

# labels_0 = ['愛知県一宮市', '千葉県市川市', '兵庫県神戸市灘区' , '山形県寒河江市', '愛知県名古屋市中川区', '兵庫県西宮市', '滋賀県近江八幡市', '愛知県犬山市', '新潟県新潟市江南区','青森県平川市']
# labels_1 = ['東京都大田区', '東京都品川区', '東京都国立市', '東京都西東京市', '東京都豊島区', '東京都足立区', '東京都青梅市']
# labels_2 = ['大阪府堺市東区', '大阪府泉大津市', '大阪府大阪市東住吉区', '大阪府東大阪市', '大阪府河内長野市', '大阪府八尾市', '大阪府高石市']
# labels_3 = ['北海道上川郡鷹栖町', '北海道旭川市', '北海道札幌市東区', '北海道滝川市', '北海道江別市', '北海道三笠市', '北海道古宇郡泊村']

# labels_0 = ['愛知県犬山市', '新潟県新潟市江南区','青森県平川市']
# labels_1 = ['東京都品川区', '東京都国立市']
# labels_1 = ['東京都文京区', '東京都日野市', '東京都杉並区']
# labels_2 = ['大阪府守口市', '大阪府吹田市',  '大阪府貝塚市']
# labels_3 = ['北海道赤平市', '北海道釧路市', '北海道苫小牧市']
labels_0 = ['茨城県高萩市','岐阜県岐阜市','滋賀県守山市']

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
# labels = [labels_1, labels_2, labels_3]
labels = [labels_0]
# labels_2 = ['北海道上川郡鷹栖町', '福岡県北九州市小倉南区','宮城県仙台市泉区', '千葉県千葉市中央区', '東京都品川区', '東京都国立市',  '大阪府堺市東区', '東京都大田区', '大阪府大阪市東住吉区','北海道旭川市', '北海道札幌市東区', '北海道滝川市','神奈川県川崎市多摩区', '神奈川県相模原市南区']
for count, label in enumerate(labels):
    for la in tqdm(label):
        for num in range(160):
            char_list = '袖印房匝瑳香芝睦柄夷隅御鋸港墨世並荒練梅調分寺狛摩あ瑞穂の出宅丈保塚模逗秦綾開箱甲蒲発附燕糸魚妙胎聖弥彦粟氷砺射善輪洲咋灘宝鳳穴敦鯖永韮杜斐笛巨早身桂忍鳴菅諏訪諸駒曲曇科県辰箕智売泰阜喬丘曽祖績埴施温'
            str_add_down = ''
            str_add_up = ''
            choice = random.randint(2,10)
            for i in range(6):
                random_char = random.choice(char_list)
                str_add_down = str_add_down + random_char
            choice = random.randint(2,10)
            for i in range(2):
                random_char = random.choice(char_list)
                str_add_up = str_add_up + random_char



            # print(num)
            path = '/home/anlab/Tienanh-backup/TrainHWJapanese/data/28062022_text_detection/dataset_TH8/dataset_8_addition_label_2007/'
            # if os.path.isdir(path)==False: os.mkdir(path)
            mask_1, x1, y1, x2, y2, size_width_1, size_height_1  = generate(la+str_add_up, num, True)
            mask_2, _, _, _, _, size_width_2, size_height_2 = generate(str_add_down, num, False)
            if size_width_1 > size_width_2:
                space = Image.new("RGB", (size_width_1-size_width_2, size_height_2), "white")
                mask_2 = get_concat_h(mask_2, space)
            else:
                space = Image.new("RGB", (size_width_2-size_width_1, size_height_1), "white")
                mask_1 = get_concat_h(mask_1, space)

            mask_concat = get_concat_v(mask_1,mask_2)
            pixel_space = 30
            white_space = Image.new("RGB", (pixel_space, mask_concat.height), "white")
            mask_concat = get_concat_h(white_space, mask_concat)
            mask_concat = get_concat_h(mask_concat, white_space)
            size_width = mask_concat.width
            size_height = mask_concat.height

            # print(x1, y1, x2, y2, size_width, size_height)
            x_hat,y_hat,w_hat,h_hat = pascal_voc_to_yolo(x1+pixel_space,y1, x2+pixel_space, y2,size_width,size_height)
            with open('/home/anlab/Tienanh-backup/TrainHWJapanese/data/28062022_text_detection/dataset_TH8/label_dataset_8_addition_label_2007/'+la+str(num) + '.txt', 'w', encoding='utf-8') as f:
                f.write(str(count)+' '+str(x_hat)+' '+str(y_hat)+' '+str(w_hat)+' '+str(h_hat))
            mask = np.array(mask_concat) 
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            # print(mask.shape)
            



            # blur
            smooth = cv2.GaussianBlur(mask, (33,33), 0)

            # divide gray by morphology image
            division = cv2.divide(mask, smooth, scale=255)



            # sharpen using unsharp masking
            sharp = filters.unsharp_mask(division, radius=1.5, amount=2.5, multichannel=False, preserve_range=False)
            sharp = (255*sharp).clip(0,255).astype(np.uint8)
            # sharp = cv2.equalizeHist(sharp)
            np.random.seed(2022)
            choice_all = random.randint(0,1)
            


            img = Image.fromarray(sharp)
            # seq = Sequence([RandomRotate(3)])
            # img, _ = seq(sharp, bboxes)

            # white = (255)
            # angle = random.uniform(-5,5)
            # img = img.rotate(angle, 0, expand = 1, fillcolor = white)

            blur = False 
            if choice > 0.6:
                img = GaussianNoise()(img, mag=0)
                img = ImpulseNoise()(img, mag=0)
            # choice = random.uniform(0,1)
            # if choice > 0.2:
            #     img = MotionBlur()(img, mag=0)
            choice = random.uniform(0,1)
            if choice > 0.7:
                img = DefocusBlur()(img, mag=0)
                blur = True
            choice = random.uniform(0,1)
            if choice > 0.8 and blur == True:
                img = Contrast()(img, mag = 1)
            choice = random.uniform(0,1)
            if choice > 0.7:
                img = Brightness()(img, mag = 0)

            img = np.array(img)
            cv2.imwrite(path+la+str(num)+'.jpg', img)
            # print(count, label)
