import sys, time
import random

def main():
	parrots = [":angelparrot:", ":aussieparrot:", ":boredparrot:", ":ceilingparrot:", 
				":coffeeparrot:", ":confusedparrot:", ":congaparrot:", ":congapartyparrot:", 
				":dealwithitparrot:", ":evilparrot:", ":fastparrot:", ":happyparrot:", 
				":icecreamparrot:", ":jediparrot:", ":margaritaparrot:", ":parrot:",
				":parrotwave1:", ":parrotwave2:", ":parrotwave3:", ":parrotwave4:", ":parrotwave5:",
				":parrotwave6:", ":pirateparrot:", ":pizzaparrot:", ":portalparrot:", ":sithparrot:",
				":thumbsupparrot:", ":ultrafastparrot:", ":wildparrot:"]

	parrots = [":congaparrot:", ":congapartyparrot:"]
	

	cmd = ""

	while(len(cmd) < 4000-8):
		cmd = cmd + (random.choice(parrots))

	print(cmd)

	with open("parrots.txt", "w") as f:
		f.write(cmd)


if __name__ == "__main__":
  main() 	