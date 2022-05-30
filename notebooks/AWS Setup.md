[YOLOv5 AWS Setup Tutorial](https://docs.ultralytics.com/environments/AWS-Quickstart/)

[EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/)
- Chose a g4ad.xlarge because it's the cheapest GPU instance available (and can always be upgraded later). Pricing is about $0.38 per hour, or $9 per day.
- Trying to request it as a spot instance because it's cheaper (up to 70%). Unfortunately, there's a limit on our vCPU spot instance requests, so I submitted a support center request.
- Currently booted up a t2.2xlarge as a spot instance

