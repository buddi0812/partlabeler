---
workflow: product-launch-video
flow: automation
storyboard: yes
message: "Turn raw factory video into a training dataset in minutes: free, local, and it runs in Colab."
destination: website
aspect: 1920x1080
language: en
audience: "ML and vision engineers, and their teammates, who label industrial parts for detector training"
length: 60s
angle: "workflow, then comparison"
narration: no
---

## Intent

A quick promo for the PartLabeler GitHub repo, sent to a few teammates for review. Sell, don't just show:
walk through the workflow (click a part to box it, find similar parts, track through the video, confirm,
export in five formats, Teach & Transfer onto similar videos), then a fair comparison against CVAT,
Label Studio, X-AnyLabeling and Roboflow Annotate, then the one-click install / Colab close.
The user asked for a "quick interactive, advertising-style video that demonstrates our project's pros over
competitors" and to show it in GitHub.

## Assets

- Pexels video 2376982 (Video Kickstarter, "A close-up video of a circuit board components", Pexels license) — PCB panel with dozens of identical units; find-similar beat.
- Pexels video 852388 (Distill, "Assembly Line", Pexels license) — rollers, bolts and a carton stack crossing the frame; click-to-box and tracking beats.
- Real PartLabeler UI captured from a demo instance running on those two clips (start screen, annotator).

## Customizations

- Silent: on-screen text only, no narration, no music (`music: none`, no SCRIPT.md).
- Deliver `renders/video.mp4` plus a short looping GIF for the top of README.md that links to the MP4
  (GitHub does not play repo video files inline). Composition source stays in `videos/partlabeler-promo/`.
- Look: PartLabeler's own tokens (teal #0a7c78, ink #15202b, light panels #f3f5f7), captured from the app.

## Notes

- NEVER use the user's private example footage, frames, labels or project names anywhere (not for public).
  Only neutral visuals and openly licensed public footage.
- Competitor claims only where a current source confirms them (research in progress); mark "as of Sep 2026",
  with sources listed in the README. Keep it fair: no claims that a competitor "can't" do something unless verified.
- Performance numbers used must be ones measured in spikes/REPORT.md, stated without naming the private example dataset.
