import os
from google import genai
from google.genai import types

def translate_srt(input_srt_path: str, output_srt_path: str, api_key: str, model_name: str = "gemini-3-pro-preview") -> tuple[bool, str]:
    """Read an SRT file, translate it to Vietnamese using Gemini, and save the output. Returns (success, error_msg)."""
    try:
        import os
        if not os.path.exists(input_srt_path):
            return False, f"Input file not found: {input_srt_path}"
            
        with open(input_srt_path, "r", encoding="utf-8") as f:
            srt_content = f.read()

        print(f"🌐 [TransAPI] Sending {len(srt_content)} chars to Gemini for translation...")
        
        api_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        if not api_keys:
            return False, "No API keys provided in settings."

        print(f"🧠 [TransAPI] Available API Keys: {len(api_keys)}. Model: {model_name}. Attempting translation...")
        
        prompt_instruction = """
        Bạn là chuyên gia dịch phụ đề Trung → Việt (ngôn tình/drama). 
Nhiệm vụ: nhận vào 1 file SRT tiếng Trung và trả ra 1 file SRT tiếng Việt.

⚠️ QUY TẮC XUẤT KẾT QUẢ (KHÓA CỨNG)
- Bạn PHẢI suy nghĩ/kiểm tra nội bộ trước khi trả lời, nhưng TUYỆT ĐỐI KHÔNG được in ra bất kỳ phần suy nghĩ nào.
- Output cuối cùng CHỈ LÀ NỘI DUNG SRT ĐÃ DỊCH. Không thêm lời mở đầu/kết luận/giải thích/bảng/ghi chú.
- TUYỆT ĐỐI KHÔNG dùng tiếng Anh (trừ khi trong SRT gốc có tiếng Anh thật sự).
- TUYỆT ĐỐI KHÔNG để sót chữ Hán/tiếng Trung trong dòng thoại đầu ra.
- KHÔNG được lặp block / không được nhân đôi subtitle.

✅ BẢO TOÀN ĐỊNH DẠNG SRT (BẮT BUỘC)
- Giữ nguyên số thứ tự của từng block.
- Giữ nguyên timecode.
- Giữ nguyên số dòng trong mỗi block nếu có thể.
- Không phá tag/ký hiệu: ♪, [..], (..), {\an8}, <i>..</i>, 【】... 
  => chỉ dịch phần chữ bên trong.

🧹 LỌC RÁC / KÝ TỰ LỖI (BẮT BUỘC)
Trong dòng thoại gốc có thể bị dính “rác OCR” như:
- chuỗi vô nghĩa: 000, 0O0o0, ww, an, R, S, X, SUUuy, NUvOy, YbUU...
- ký tự đơn lẻ/nhóm ngẫu nhiên: 改, 营鑫新, 邮楼, 楼, 囍, 福, 发, 中...
Quy tắc xử lý:
- Nếu nó KHÔNG góp nghĩa cho lời thoại: XÓA khỏi bản dịch.
- Nếu nó là tag/markup thật sự (ví dụ {\an8} hoặc <i>): GIỮ.
- Nếu nó có vẻ là tên riêng thật sự hoặc thuật ngữ có nghĩa: dịch/phiên âm Hán Việt theo quy tắc tên riêng bên dưới.

🎭 QUY TẮC DỊCH THOẠI (PHỤ ĐỀ ĐỌC NHANH)
- Dịch tự nhiên, khẩu ngữ phim, câu ngắn.
- Không tự ý thêm nội dung.
- Lời khịa/chửi: mức “phim Việt”, không quá tục nhưng đủ thái độ.

🧑‍🤝‍🧑 XƯNG HÔ (ƯU TIÊN AN TOÀN)
- Nếu thiếu ngữ cảnh: dùng “tôi / anh / cô / cậu” trung tính và giữ nhất quán toàn file.
- 哥/姐/弟/妹: anh/chị/em.

🈶 TÊN RIÊNG & THUẬT NGỮ (HÁN VIỆT + NHẤT QUÁN)
- TẤT CẢ tên người/địa danh/môn phái/chức vị/vật phẩm: chuyển sang Hán Việt.
- Ví dụ gặp họ tên Trung: 赵黑龙 = Triệu Hắc Long, 赵青衣 = Triệu Thanh Y, 唐城 = Đường Thành.
- Nếu một tên xuất hiện nhiều lần: dùng đúng 1 cách viết xuyên suốt.
- Nếu không chắc Hán Việt: giữ nguyên chữ Hán cho đúng, không bịa.

🧪 BƯỚC TỰ KIỂM TRA (BẮT BUỘC TRƯỚC KHI TRẢ OUTPUT)
Trước khi xuất:
1) Quét toàn bộ output: không được còn chữ Hán .
2) Quét toàn bộ output: không được có tiếng Anh lạ.
3) Không còn “rác OCR” kiểu 000/ww/an/R/改/营鑫新... trong dòng thoại.
4) Không lặp block; số thứ tự và timecode giữ nguyên.

Bây giờ, đây là SRT cần dịch (dưới đây). Hãy trả về DUY NHẤT SRT đã dịch:

"""
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt_instruction + srt_content)],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.2, # Low temperature for more deterministic/stable layout handling
        )
        
        last_error = None
        for i, current_key in enumerate(api_keys):
            try:
                masked_key = current_key[:5] + "..." + current_key[-4:] if len(current_key) > 10 else current_key
                print(f"📡 [TransAPI] Trying Key #{i+1} ({masked_key})...")
                
                client = genai.Client(api_key=current_key)
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=generate_content_config,
                )
                
                # If we get here without an exception, it succeeded!
                print(f"✅ [TransAPI] Translation successful using Key #{i+1}!")
                translated_text = response.text.strip()
                break # exit the retry loop
                
            except Exception as api_err:
                print(f"⚠️ [TransAPI] Key #{i+1} failed: {api_err}")
                last_error = api_err
                
        else: # This 'else' runs if the for-loop never 'break'ed (i.e., all keys failed)
            err_msg = str(last_error) if last_error else "Unknown error."
            print(f"❌ [TransAPI] All {len(api_keys)} available API keys failed! Last error: {err_msg}")
            return False, f"All keys failed. Last error: {err_msg}"
            
        # Remove any markdown code block wrappers if Gemini accidentally adds them
        if translated_text.startswith("```"):
            lines = translated_text.split('\n')
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].startswith("```"): lines = lines[:-1]
            translated_text = '\n'.join(lines).strip()
            
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
            
        print(f"💾 [TransAPI] Saved translated SRT to: {output_srt_path}")
        return True, ""
        
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        print(f"❌ [TransAPI] Fatal Translation Error:\n{err_str}")
        return False, str(e)

# For direct testing via terminal
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python trans_api.py <input.srt> <output.vi.srt> <api_key,api_key...> [model_name]")
        print("Example: python trans_api.py episode.srt episide_vi.srt AIzaSy... gemini-3-pro-preview")
    else:
        model_name = sys.argv[4] if len(sys.argv) > 4 else "gemini-3-pro-preview"
        translate_srt(sys.argv[1], sys.argv[2], sys.argv[3], model_name)
