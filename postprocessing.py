# ==========================================================================================================================================================
# metrics postprocessing 
# purpose: postprocess metrics data in csv to generate visualisations
# ==========================================================================================================================================================

import os
import shutil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# marl_train_test options
agent_model 												= "maddpg" 
adver_model 												= "maddpg"
agent_mode 													= "train"
adver_mode 													= "train"
general_training_name 										= "agent_" + AGENT_MODEL + "_vs_opp_"  + ADVER_MODEL + "_1"
csv_log_directory											= "csv_log"

def post_process(csv_log_directory, general_training_name, agent_mode, adver_mode):

	# extract csv files into pandas dataframes
	metrics = pd.read_csv(csv_log_directory + '/' + general_training_name + "_logs.csv", header = 0)
	
	# make directory for plots, if is already exists, override it
	try:

		os.mkdir("training_plots/" + general_training_name)

	except:

		shutil.rmtree("training_plots/" + general_training_name)
		os.mkdir("training_plots/" + general_training_name)

	# frequency of success plot
	plt.title("Frequency of Success")
	data = [metrics['sum_agent_wins'][-1], metrics['sum_adver_wins'][-1]]
	plot_1 = plt.bar(x = np.arange(len(data)), height = data, tick_label = ['agent', 'adversary'])
	plt.ylabel("Frequency")
	plt.xlabel("Drone type")
	plt.savefig("training_plots/" + general_training_name + "/" + general_training_name + "_freq_success.pdf", 
				bbox_inches = 'tight')
	plt.close()

	# frequency of exceeding screen plot
	plt.title("Frequency of Exceeding Screen")
	data = [metrics['sum_agent_exceed_screen'][-1], metrics['sum_adver_exceed_screen'][-1]]
	plot_2 = plt.bar(x = np.arange(len(data)), height = data, tick_label = ['agent', 'adversary'])
	plt.ylabel("Frequency")
	plt.xlabel("Drone type")
	plt.savefig("training_plots/" + general_training_name + "/" + general_training_name + "_freq_exceed_screen.pdf", 
				bbox_inches = 'tight')
	plt.close()

	# sum_agent_number_of_team_collisions plot
	plt.title("Sum of Agent Team Collisions vs Number of Episodes")
	plot_3 = sns.lineplot(data = np.array(metrics['sum_agent_number_of_team_collisions']))
	plt.ylabel("Collisions")
	plt.xlabel("Number of episodes")
	plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "sum_agent_team_collisions_vs_episodes.pdf", 
				bbox_inches = 'tight')
	plt.close()

	# sum_agent_number_of_team_collisions plot
	plt.title("Sum of Agent Opponent Collisions vs Number of Episodes")
	plot_4 = sns.lineplot(data = np.array(metrics['sum_agent_number_of_oppo_collisions']))
	plt.ylabel("Collisions")
	plt.xlabel("Number of episodes")
	plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "sum_agent_oppo_collisions_vs_episodes.pdf", 
				bbox_inches = 'tight')
	plt.close()

	# sum_adver_number_of_team_collisions plot
	plt.title("Sum of Adversarial Team Collisions vs Number of Episodes")
	plot_5 = sns.lineplot(data = np.array(metrics['sum_adver_number_of_team_collisions']))
	plt.ylabel("Collisions")
	plt.xlabel("Number of episodes")
	plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "sum_adver_team_collisions_vs_episodes.pdf", 
				bbox_inches = 'tight')
	plt.close()

	# sum_adver_number_of_team_collisions plot
	plt.title("Sum of Adversarial Opponent Collisions vs Number of Episodes")
	plot_6 = sns.lineplot(data = np.array(metrics['sum_adver_number_of_oppo_collisions']))
	plt.ylabel("Collisions")
	plt.xlabel("Number of episodes")
	plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "sum_adver_oppo_collisions_vs_episodes.pdf", 
				bbox_inches = 'tight')
	plt.close()

	# avg_agent_number_of_team_collisions plot
	plt.title("Average Agent Team Collisions vs Number of Episodes")
	plot_7 = sns.lineplot(data = np.array(metrics['avg_agent_number_of_team_collisions']))
	plt.ylabel("Collisions")
	plt.xlabel("Number of episodes")
	plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "avg_agent_team_collisions_vs_episodes.pdf", 
				bbox_inches = 'tight')
	plt.close()

	# avg_agent_number_of_team_collisions plot
	plt.title("Average Agent Opponent Collisions vs Number of Episodes")
	plot_8 = sns.lineplot(data = np.array(metrics['avg_agent_number_of_oppo_collisions']))
	plt.ylabel("Collisions")
	plt.xlabel("Number of episodes")
	plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "avg_agent_oppo_collisions_vs_episodes.pdf", 
				bbox_inches = 'tight')
	plt.close()

	# avg_adver_number_of_team_collisions plot
	plt.title("Average Adversarial Team Collisions vs Number of Episodes")
	plot_9 = sns.lineplot(data = np.array(metrics['avg_adver_number_of_team_collisions']))
	plt.ylabel("Collisions")
	plt.xlabel("Number of episodes")
	plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "avg_adver_team_collisions_vs_episodes.pdf", 
				bbox_inches = 'tight')
	plt.close()

	# avg_adver_number_of_team_collisions plot
	plt.title("Average Adversarial Opponent Collisions vs Number of Episodes")
	plot_10 = sns.lineplot(data = np.array(metrics['avg_adver_number_of_oppo_collisions']))
	plt.ylabel("Collisions")
	plt.xlabel("Number of episodes")
	plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "avg_adver_oppo_collisions_vs_episodes.pdf", 
				bbox_inches = 'tight')
	plt.close()

	# check that agent is training
	if agent_mode != 'test':

		# avg_agent_actor_loss plot
		plt.title("Average Agent Actor Loss vs Number of Episodes")
		plot_11 = sns.lineplot(data = np.array(metrics['avg_agent_actor_loss']))
		plt.ylabel("Loss")
		plt.xlabel("Number of episodes")
		plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "avg_agent_actor_loss_vs_episodes.pdf", 
					bbox_inches = 'tight')
		plt.close()

		# avg_agent_critic_loss plot
		plt.title("Average Agent Critic Loss vs Number of Episodes")
		plot_12 = sns.lineplot(data = np.array(metrics['avg_agent_critic_loss']))
		plt.ylabel("Loss")
		plt.xlabel("Number of episodes")
		plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "avg_agent_critic_loss_vs_episodes.pdf", 
					bbox_inches = 'tight')
		plt.close()

	# check that adversarial is training
	if adver_mode != 'test':

		# avg_adver_actor_loss plot
		plt.title("Average Adversarial Actor Loss vs Number of Episodes")
		plot_11 = sns.lineplot(data = np.array(metrics['avg_adver_actor_loss']))
		plt.ylabel("Loss")
		plt.xlabel("Number of episodes")
		plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "avg_adver_actor_loss_vs_episodes.pdf", 
					bbox_inches = 'tight')
		plt.close()

		# avg_adver_critic_loss plot
		plt.title("Average Adversarial Critic Loss vs Number of Episodes")
		plot_12 = sns.lineplot(data = np.array(metrics['avg_adver_critic_loss']))
		plt.ylabel("Loss")
		plt.xlabel("Number of episodes")
		plt.savefig("training_plots/" + general_training_name  + "/" + general_training_name + "avg_adver_critic_loss_vs_episodes.pdf", 
					bbox_inches = 'tight')
		plt.close()

if __name__ == "__main__":

	# conduct the post processing for the case when this script is called manually
	post_process(csv_log_directory, general_training_name, agent_mode, adver_mode)