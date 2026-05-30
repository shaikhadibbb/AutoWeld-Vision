from pathlib import Path
from PIL import Image, ImageDraw


def draw_rounded_rect(draw, coords, radius, color):
    """Draws a rounded rectangle."""
    x1, y1, x2, y2 = coords
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=color)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=color)
    draw.pieslice([x1, y1, x1 + radius * 2, y1 + radius * 2], 180, 270, fill=color)
    draw.pieslice([x2 - radius * 2, y1, x2, y1 + radius * 2], 270, 360, fill=color)
    draw.pieslice([x1, y2 - radius * 2, x1 + radius * 2, y2], 90, 180, fill=color)
    draw.pieslice([x2 - radius * 2, y2 - radius * 2, x2, y2], 0, 90, fill=color)


def generate_screenshots():
    """Generates 4 high-fidelity mock screenshots of the Streamlit App."""
    width, height = 800, 500
    frames = []

    # Base background (Dark Mode streamlit theme)
    bg_color = (14, 17, 23)  # Streamlit dark background
    sidebar_color = (38, 39, 48)  # Sidebar dark grey
    text_color = (250, 250, 250)  # Near white
    accent_color = (255, 75, 75)  # Streamlit Red
    card_bg = (29, 31, 41)  # Card background

    # Create 4 frames
    for i in range(4):
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw Sidebar
        draw.rectangle([0, 0, 220, height], fill=sidebar_color)
        draw.text((20, 30), "🔧 AutoWeld-Vision", fill=text_color)
        draw.text((20, 60), "---", fill=(100, 100, 100))
        draw.text((20, 90), "Inspection Settings", fill=(180, 180, 180))
        draw.text((20, 120), "VIN: BMW-G60-2026", fill=(150, 150, 150))

        # Sidebar Status
        draw.rectangle([20, 380, 200, 470], fill=(20, 20, 30))
        draw.text((30, 390), "System: Active 🟢", fill=(0, 255, 0))
        draw.text((30, 420), "Mode: Unsupervised", fill=(150, 150, 150))

        # Main Panel Header
        draw.text((250, 30), "🔧 AutoWeld-Vision: Operator Terminal", fill=text_color)
        draw.text(
            (250, 60),
            "Real-time weld seam anomaly screening and visual audit log generator.",
            fill=(150, 150, 150),
        )
        draw.rectangle([250, 90, 780, 92], fill=(80, 80, 80))

        if i == 0:
            # Frame 1: Awaiting Upload
            draw.text(
                (250, 120),
                "Step 1: Upload a weld seam image or use default sample",
                fill=text_color,
            )
            draw_rounded_rect(draw, [250, 160, 750, 320], 10, card_bg)
            draw.text(
                (420, 230), "📁 Drag and drop weld image here", fill=(150, 150, 150)
            )
            draw.text((460, 260), "Limit 200MB • PNG, JPG, JPEG", fill=(100, 100, 100))

            # Big blue button mockup
            draw_rounded_rect(draw, [250, 350, 500, 400], 5, accent_color)
            draw.text((310, 365), "🚀 Execute Visual Audit Seam", fill=text_color)

        elif i == 1:
            # Frame 2: Model Loading / Running
            draw.text(
                (250, 120),
                "Step 1: Upload a weld seam image or use default sample",
                fill=(150, 150, 150),
            )
            draw_rounded_rect(draw, [250, 160, 750, 320], 10, card_bg)
            draw.text(
                (360, 220),
                "⌛ Running Late-Fusion Coreset Optimization...",
                fill=accent_color,
            )

            # Progress bar
            draw_rounded_rect(draw, [320, 260, 680, 275], 5, (50, 50, 50))
            draw_rounded_rect(draw, [320, 260, 550, 275], 5, accent_color)

            draw_rounded_rect(draw, [250, 350, 500, 400], 5, (100, 100, 100))
            draw.text((310, 365), "Processing Tensors...", fill=(200, 200, 200))

        elif i == 2:
            # Frame 3: Inspection Results - Heatmap Overlay
            draw.text(
                (250, 110),
                "Quality Inspection Complete! 🔴 Anomaly Detected",
                fill=(255, 75, 75),
            )

            # Metrics cards
            draw_rounded_rect(draw, [250, 140, 400, 200], 5, card_bg)
            draw.text((260, 150), "Decision", fill=(150, 150, 150))
            draw.text((260, 170), "FAIL", fill=(255, 75, 75))

            draw_rounded_rect(draw, [420, 140, 570, 200], 5, card_bg)
            draw.text((430, 150), "Anomaly Score", fill=(150, 150, 150))
            draw.text((430, 170), "0.8966", fill=accent_color)

            draw_rounded_rect(draw, [590, 140, 750, 200], 5, card_bg)
            draw.text((600, 150), "Model Mode", fill=(150, 150, 150))
            draw.text((600, 170), "ENSEMBLE (REAL)", fill=(0, 255, 0))

            # Side by side mock previews
            draw_rounded_rect(draw, [250, 220, 490, 450], 5, card_bg)
            draw.text((260, 230), "Original Seam", fill=(150, 150, 150))
            # Simulated weld line (curved white brush)
            draw.arc([270, 280, 470, 420], 180, 360, fill=(200, 200, 200), width=15)

            draw_rounded_rect(draw, [510, 220, 750, 450], 5, card_bg)
            draw.text((520, 230), "Heatmap Overlay", fill=(150, 150, 150))
            # Simulated weld line with red/green anomalies
            draw.arc([530, 280, 730, 420], 180, 360, fill=(0, 200, 0), width=15)
            # Defect red overlay circle
            draw.ellipse([610, 270, 650, 310], fill=(255, 0, 0, 120))
            draw.text((615, 280), "Pore", fill=(255, 255, 255))

        elif i == 3:
            # Frame 4: Quality Report Certified & Download
            draw.text(
                (250, 110),
                "Quality Inspection Complete! 🔴 Anomaly Detected",
                fill=(255, 75, 75),
            )

            # Large Certificate Mockup
            draw_rounded_rect(draw, [250, 150, 750, 380], 10, card_bg)
            draw.text(
                (320, 180),
                "★ IATF 16949 QUALITY ASSURANCE AUDIT RECORD ★",
                fill=(255, 215, 0),
            )
            draw.text((280, 220), "Chassis VIN:  BMW-G60-2026", fill=text_color)
            draw.text(
                (280, 255), "Timestamp:    2026-05-31 00:06:12 UTC", fill=text_color
            )
            draw.text(
                (280, 290),
                "Inspection:  FAIL  |  Core Score: 0.8966  |  Limit: 0.50",
                fill=text_color,
            )
            draw.text(
                (280, 325),
                "Certified Secure Visual Audit Hash: sha256_e10dced...",
                fill=(120, 120, 120),
            )

            # Download Button
            draw_rounded_rect(draw, [250, 410, 750, 460], 5, (0, 200, 100))
            draw.text(
                (430, 425), "📥 Download Certified Quality PDF Report", fill=text_color
            )

        frames.append(img)

    # Save GIFs
    Path("demo").mkdir(exist_ok=True)
    for gif_name in ["demo_run.gif", "demo_final.gif"]:
        frames[0].save(
            f"demo/{gif_name}",
            save_all=True,
            append_images=frames[1:],
            duration=1250,  # 1.25s per frame = 5 seconds total for 4 frames
            loop=0,
        )
    print(
        "✓ Successfully generated looping GIFs at demo/demo_run.gif and demo/demo_final.gif"
    )


if __name__ == "__main__":
    generate_screenshots()
