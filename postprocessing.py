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

# postprocessing options
NUMBER_OF_EPISODES											= 1000	
CSV_LOG_DIRECTORY											= "csv_log"
POSTPROCESSING_DIRECTORY_NAME								= "test_maddpgv2_3_vs_3_1_big_5_seconds_plots"
LEGEND_OUT													= True
TRAINING_LIST												= [[["maddpgv2", "maddpgv2"], [3, 3], ["1_big_5_sec"], ["test", "test"]]]
# TRAINING_LIST												= [[["maddpg", "maddpg"], [1, 1], ["1_big_5_sec"], ["test", "test"]], [["mappo", "mappo"], [1, 1], ["1_big_5_sec"], ["test", "test"]], 
# 															   [["maddpgv2", "maddpgv2"], [1, 1], ["1_big_5_sec"], ["test", "test"]]]

def post_process(number_of_episodes, csv_log_directory, postprocessing_directory_name, training_list):

	""" function to post process csv logs to generate visulisations for metrics """

	# list to store metrics
	metrics_list = []

	# list to store versus labels
	vs_labels_list = []

	# iterate over length of training list
	for i in range(len(training_list)):

		# obtain agent and adver model
		agent_model = training_list[i][0][0]
		adver_model = training_list[i][0][1]

		# obtain number of agent and adversarial drones
		num_of_agent = training_list[i][1][0]
		num_of_adver = training_list[i][1][1]

		# obtain additional training name
		add_training_name_str = training_list[i][2][0]

		# obtain general training name
		general_training_name = "agent_" + agent_model + "_vs_opp_"  + adver_model + "_" + str(num_of_agent) + "_vs_" + str(num_of_adver) + "_" + add_training_name_str

		# append pandas data frame to metrics list
		metrics_list.append(pd.read_csv(csv_log_directory + '/' + general_training_name + "_" + training_list[i][3][0] + "_" + training_list[i][3][1] + "_logs.csv", header = 0))

		# append vs labels
		vs_labels_list.append(agent_model.upper() + " v " + adver_model.upper())
	
	# make directory for plots, if is already exists, override it
	try:

		os.mkdir("training_plots/" + postprocessing_directory_name)

	except:

		shutil.rmtree("training_plots/" + postprocessing_directory_name)
		os.mkdir("training_plots/" + postprocessing_directory_name)

	# set default theme for seaborn
	sns.set_theme()

	# number_of_wins_vs_algorithms barplot
	
	# obtain data
	df1 = pd.DataFrame([{'Number of Wins': metrics_list[i]['sum_agent_wins'].iloc[number_of_episodes - 1], 'Drone Type': 'Agent', 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	df2 = pd.DataFrame([{'Number of Wins': metrics_list[i]['sum_adver_wins'].iloc[number_of_episodes - 1], 'Drone Type': 'Adver', 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	data = pd.concat([df1, df2])

	# plot and save figure
	plot = sns.barplot(x = "Algorithms", y = "Number of Wins", hue = "Drone Type", data = data)
	plot.set_title("Number of Wins vs Algorithms")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/number_of_wins_vs_algorithms_barplot.pdf", bbox_inches = 'tight')
	plt.close()

	# number_of_wins_vs_number_of_episodes lineplot
	
	# obtain data 
	data = pd.DataFrame()

	for i in range(len(metrics_list)):

		df1 = pd.DataFrame([{'Number of Wins': metrics_list[i]['sum_agent_wins'].iloc[ep], 'Drone Type': 'Agent', 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 
							 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		df2 = pd.DataFrame([{'Number of Wins': metrics_list[i]['sum_adver_wins'].iloc[ep], 'Drone Type': 'Adver', 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 
							 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		data = pd.concat([data, df1, df2])
	
	# plot and save figure
	hue = data['Algorithms'].astype(str) + ', ' + data['Drone Type'].astype(str)
	plot = sns.lineplot(x = "Number of Episodes", y = "Number of Wins", hue = hue, data = data)
	plot.set_title("Number of Wins vs Number of Episodes")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/number_of_wins_vs_number_of_episodes_lineplot.pdf", bbox_inches = 'tight')
	plt.close()

	# number_of_screen_exits_vs_algorithms barplot
	
	# obtain data
	df1 = pd.DataFrame([{'Number of Screen Exits': metrics_list[i]['sum_agent_exceed_screen'].iloc[number_of_episodes - 1], 'Drone Type': 'Agent', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	df2 = pd.DataFrame([{'Number of Screen Exits': metrics_list[i]['sum_adver_exceed_screen'].iloc[number_of_episodes - 1], 'Drone Type': 'Adver', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	data = pd.concat([df1, df2])

	# plot and save figure
	plot = sns.barplot(x = "Algorithms", y = "Number of Screen Exits", hue = "Drone Type", data = data)
	plot.set_title("Number of Screen Exits vs Algorithms")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/number_of_screen_exits_vs_algorithms_barplot.pdf", bbox_inches = 'tight')
	plt.close()

	# number_of_screen_exits_vs_number_of_episodes lineplot

	# obtain data 
	data = pd.DataFrame()

	for i in range(len(metrics_list)):

		df1 = pd.DataFrame([{'Number of Screen Exits': metrics_list[i]['sum_agent_exceed_screen'].iloc[ep], 'Drone Type': 'Agent', 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 
							 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		df2 = pd.DataFrame([{'Number of Screen Exits': metrics_list[i]['sum_adver_exceed_screen'].iloc[ep], 'Drone Type': 'Adver', 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 
							 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		data = pd.concat([data, df1, df2])
	
	# plot and save figure
	hue = data['Algorithms'].astype(str) + ', ' + data['Drone Type'].astype(str)
	plot = sns.lineplot(x = "Number of Episodes", y = "Number of Screen Exits", hue = hue, data = data)
	plot.set_title("Number of Screen Exits vs Number of Episodes")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/number_of_screen_exits_vs_number_of_episodes_lineplot.pdf", bbox_inches = 'tight')
	plt.close()

	# number_of_team_collisions_vs_algorithms barplot
	
	# obtain data
	df1 = pd.DataFrame([{'Number of Team Collisions': metrics_list[i]['sum_agent_number_of_team_collisions'].sum(), 'Drone Type': 'Agent', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	df2 = pd.DataFrame([{'Number of Team Collisions': metrics_list[i]['sum_adver_number_of_team_collisions'].sum(), 'Drone Type': 'Adver', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	data = pd.concat([df1, df2])

	# plot and save figure
	plot = sns.barplot(x = "Algorithms", y = "Number of Team Collisions", hue = "Drone Type", data = data)
	plot.set_title("Number of Team Collisions vs Algorithms")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/number_of_team_collisions_vs_algorithms_barplot.pdf", bbox_inches = 'tight')
	plt.close()

	# number_of_team_collisions_vs_number_of_episodes lineplot

	# obtain data 
	data = pd.DataFrame()

	for i in range(len(metrics_list)):

		df1 = pd.DataFrame([{'Number of Team Collisions': metrics_list[i]['sum_agent_number_of_team_collisions'].iloc[ep], 'Drone Type': 'Agent', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		df2 = pd.DataFrame([{'Number of Team Collisions': metrics_list[i]['sum_adver_number_of_team_collisions'].iloc[ep], 'Drone Type': 'Adver', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		data = pd.concat([data, df1, df2])
	
	# plot and save figure
	hue = data['Algorithms'].astype(str) + ', ' + data['Drone Type'].astype(str)
	plot = sns.lineplot(x = "Number of Episodes", y = "Number of Team Collisions", hue = hue, data = data)
	plot.set_title("Number of Team Collisions vs Number of Episodes")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/number_of_team_collisions_vs_number_of_episodes_lineplot.pdf", bbox_inches = 'tight')
	plt.close()

	# number_of_oppo_collisions_vs_algorithms barplot
	
	# obtain data
	df1 = pd.DataFrame([{'Number of Opponent Collisions': metrics_list[i]['sum_agent_number_of_oppo_collisions'].sum(), 'Drone Type': 'Agent', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	df2 = pd.DataFrame([{'Number of Opponent Collisions': metrics_list[i]['sum_adver_number_of_oppo_collisions'].sum(), 'Drone Type': 'Adver', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	data = pd.concat([df1, df2])

	# plot and save figure
	plot = sns.barplot(x = "Algorithms", y = "Number of Opponent Collisions", hue = "Drone Type", data = data)
	plot.set_title("Number of Opponent Collisions vs Algorithms")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/number_of_oppo_collisions_vs_algorithms_barplot.pdf", bbox_inches = 'tight')
	plt.close()

	# number_of_oppo_collisions_vs_number_of_episodes lineplot

	# obtain data 
	data = pd.DataFrame()

	for i in range(len(metrics_list)):

		df1 = pd.DataFrame([{'Number of Opponent Collisions': metrics_list[i]['sum_agent_number_of_oppo_collisions'].iloc[ep], 'Drone Type': 'Agent', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		df2 = pd.DataFrame([{'Number of Opponent Collisions': metrics_list[i]['sum_adver_number_of_oppo_collisions'].iloc[ep], 'Drone Type': 'Adver', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		data = pd.concat([data, df1, df2])
	
	# plot and save figure
	hue = data['Algorithms'].astype(str) + ', ' + data['Drone Type'].astype(str)
	plot = sns.lineplot(x = "Number of Episodes", y = "Number of Opponent Collisions", hue = hue, data = data)
	plot.set_title("Number of Opponent Collisions vs Number of Episodes")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/number_of_oppo_collisions_vs_number_of_episodes_lineplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_number_of_team_collisions_vs_algorithms barplot
	
	# obtain data
	df1 = pd.DataFrame([{'Average Number of Team Collisions': metrics_list[i]['avg_agent_number_of_team_collisions'].sum(), 'Drone Type': 'Agent', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	df2 = pd.DataFrame([{'Average Number of Team Collisions': metrics_list[i]['avg_adver_number_of_team_collisions'].sum(), 'Drone Type': 'Adver', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	data = pd.concat([df1, df2])

	# plot and save figure
	plot = sns.barplot(x = "Algorithms", y = "Average Number of Team Collisions", hue = "Drone Type", data = data)
	plot.set_title("Average Number of Team Collisions vs Algorithms")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_number_of_team_collisions_vs_algorithms_barplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_number_of_team_collisions_vs_number_of_episodes lineplot

	# obtain data 
	data = pd.DataFrame()

	for i in range(len(metrics_list)):

		df1 = pd.DataFrame([{'Average Number of Team Collisions': metrics_list[i]['avg_agent_number_of_team_collisions'].iloc[ep], 'Drone Type': 'Agent', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		df2 = pd.DataFrame([{'Average Number of Team Collisions': metrics_list[i]['avg_adver_number_of_team_collisions'].iloc[ep], 'Drone Type': 'Adver', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		data = pd.concat([data, df1, df2])
	
	# plot and save figure
	hue = data['Algorithms'].astype(str) + ', ' + data['Drone Type'].astype(str)
	plot = sns.lineplot(x = "Number of Episodes", y = "Average Number of Team Collisions", hue = hue, data = data)
	plot.set_title("Average Number of Team Collisions vs Number of Episodes")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_number_of_team_collisions_vs_number_of_episodes_lineplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_number_of_oppo_collisions_vs_algorithms barplot
	
	# obtain data
	df1 = pd.DataFrame([{'Average Number of Opponent Collisions': metrics_list[i]['avg_agent_number_of_oppo_collisions'].sum(), 'Drone Type': 'Agent', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	df2 = pd.DataFrame([{'Average Number of Opponent Collisions': metrics_list[i]['avg_adver_number_of_oppo_collisions'].sum(), 'Drone Type': 'Adver', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	data = pd.concat([df1, df2])

	# plot and save figure
	plot = sns.barplot(x = "Algorithms", y = "Average Number of Opponent Collisions", hue = "Drone Type", data = data)
	plot.set_title("Average Number of Opponent Collisions vs Algorithms")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_number_of_oppo_collisions_vs_algorithms_barplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_number_of_oppo_collisions_vs_number_of_episodes lineplot

	# obtain data 
	data = pd.DataFrame()

	for i in range(len(metrics_list)):

		df1 = pd.DataFrame([{'Average Number of Opponent Collisions': metrics_list[i]['avg_agent_number_of_oppo_collisions'].iloc[ep], 'Drone Type': 'Agent', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		df2 = pd.DataFrame([{'Average Number of Opponent Collisions': metrics_list[i]['avg_adver_number_of_oppo_collisions'].iloc[ep], 'Drone Type': 'Adver', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		data = pd.concat([data, df1, df2])
	
	# plot and save figure
	hue = data['Algorithms'].astype(str) + ', ' + data['Drone Type'].astype(str)
	plot = sns.lineplot(x = "Number of Episodes", y = "Average Number of Opponent Collisions", hue = hue, data = data)
	plot.set_title("Average Number of Opponent Collisions vs Number of Episodes")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_number_of_oppo_collisions_vs_number_of_episodes_lineplot.pdf", bbox_inches = 'tight')
	plt.close()

	# elo_vs_algorithms barplot
	
	# obtain data
	df1 = pd.DataFrame([{'Elo': metrics_list[i]['agent_elo'].iloc[number_of_episodes - 1], 'Drone Type': 'Agent', 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	df2 = pd.DataFrame([{'Elo': metrics_list[i]['adver_elo'].iloc[number_of_episodes - 1], 'Drone Type': 'Adver', 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	data = pd.concat([df1, df2])

	# plot and save figure
	plot = sns.barplot(x = "Algorithms", y = "Elo", hue = "Drone Type", data = data)
	plot.set_title("Elo vs Algorithms")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/elo_vs_algorithms_barplot.pdf", bbox_inches = 'tight')
	plt.close()

	# elo_vs_number_of_episodes lineplot

	# obtain data 
	data = pd.DataFrame()

	for i in range(len(metrics_list)):

		df1 = pd.DataFrame([{'Elo': metrics_list[i]['agent_elo'].iloc[ep], 'Drone Type': 'Agent', 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 
							 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		df2 = pd.DataFrame([{'Elo': metrics_list[i]['adver_elo'].iloc[ep], 'Drone Type': 'Adver', 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 
							 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		data = pd.concat([data, df1, df2])
	
	# plot and save figure
	hue = data['Algorithms'].astype(str) + ', ' + data['Drone Type'].astype(str)
	plot = sns.lineplot(x = "Number of Episodes", y = "Elo", hue = hue, data = data)
	plot.set_title("Elo vs Number of Episodes")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/elo_vs_number_of_episodes_lineplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_actor_loss_vs_algorithms barplot

	# obtain data
	df1 = pd.DataFrame([{'Average Actor Loss': metrics_list[i]['avg_agent_actor_loss'].iloc[number_of_episodes - 1] if training_list[i][3][0] != 'test' else 0, 'Drone Type': 'Agent', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	df2 = pd.DataFrame([{'Average Actor Loss': metrics_list[i]['avg_adver_actor_loss'].iloc[number_of_episodes - 1] if training_list[i][3][1] != 'test' else 0, 'Drone Type': 'Adver', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	data = pd.concat([df1, df2])

	# plot and save figure
	plot = sns.barplot(x = "Algorithms", y = "Average Actor Loss", hue = "Drone Type", data = data)
	plot.set_title("Average Actor Loss vs Algorithms")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_actor_loss_vs_algorithms_barplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_actor_loss_vs_number_of_episodes lineplot

	# obtain data 
	data = pd.DataFrame()

	for i in range(len(metrics_list)):

		df1 = pd.DataFrame([{'Average Actor Loss': metrics_list[i]['avg_agent_actor_loss'].iloc[ep] if training_list[i][3][0] != 'test' else 0, 'Drone Type': 'Agent', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		df2 = pd.DataFrame([{'Average Actor Loss': metrics_list[i]['avg_adver_actor_loss'].iloc[ep] if training_list[i][3][1] != 'test' else 0, 'Drone Type': 'Adver', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		data = pd.concat([data, df1, df2])
	
	# plot and save figure
	hue = data['Algorithms'].astype(str) + ', ' + data['Drone Type'].astype(str)
	plot = sns.lineplot(x = "Number of Episodes", y = "Average Actor Loss", hue = hue, data = data)
	plot.set_title("Average Actor Loss vs Number of Episodes")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_actor_loss_vs_number_of_episodes_lineplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_critic_loss_vs_algorithms barplot

	# obtain data
	df1 = pd.DataFrame([{'Average Critic Loss': metrics_list[i]['avg_agent_critic_loss'].iloc[number_of_episodes - 1] if training_list[i][3][0] != 'test' else 0, 'Drone Type': 'Agent', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	df2 = pd.DataFrame([{'Average Critic Loss': metrics_list[i]['avg_adver_critic_loss'].iloc[number_of_episodes - 1] if training_list[i][3][1] != 'test' else 0, 'Drone Type': 'Adver', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	data = pd.concat([df1, df2])

	# plot and save figure
	plot = sns.barplot(x = "Algorithms", y = "Average Critic Loss", hue = "Drone Type", data = data)
	plot.set_title("Average Critic Loss vs Algorithms")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_critic_loss_vs_algorithms_barplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_critic_loss_vs_number_of_episodes lineplot

	# obtain data 
	data = pd.DataFrame()

	for i in range(len(metrics_list)):

		df1 = pd.DataFrame([{'Average Critic Loss': metrics_list[i]['avg_agent_critic_loss'].iloc[ep] if training_list[i][3][0] != 'test' else 0, 'Drone Type': 'Agent', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		df2 = pd.DataFrame([{'Average Critic Loss': metrics_list[i]['avg_adver_critic_loss'].iloc[ep] if training_list[i][3][1] != 'test' else 0, 'Drone Type': 'Adver', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		data = pd.concat([data, df1, df2])
	
	# plot and save figure
	hue = data['Algorithms'].astype(str) + ', ' + data['Drone Type'].astype(str)
	plot = sns.lineplot(x = "Number of Episodes", y = "Average Critic Loss", hue = hue, data = data)
	plot.set_title("Average Critic Loss vs Number of Episodes")
	if LEGEND_OUT == True:
		plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_critic_loss_vs_number_of_episodes_lineplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_actor_grad_norm_vs_algorithms barplot

	# obtain data
	df1 = pd.DataFrame([{'Average Actor Gradient Norm': metrics_list[i]['avg_agent_actor_grad_norm'].iloc[number_of_episodes] if training_list[i][3][0] != 'test' else 0, 'Drone Type': 'Agent', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	df2 = pd.DataFrame([{'Average Actor Gradient Norm': metrics_list[i]['avg_adver_actor_grad_norm'].iloc[number_of_episodes] if training_list[i][3][1] != 'test' else 0, 'Drone Type': 'Adver', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	data = pd.concat([df1, df2])

	# plot and save figure
	plot = sns.barplot(x = "Algorithms", y = "Average Actor Gradient Norm", hue = "Drone Type", data = data)
	plot.set_title("Average Actor Gradient Norm vs Algorithms")
	plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_actor_grad_norm_vs_algorithms_barplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_actor_grad_norm_vs_number_of_episodes lineplot

	# obtain data 
	data = pd.DataFrame()

	for i in range(len(metrics_list)):

		df1 = pd.DataFrame([{'Average Actor Gradient Norm': metrics_list[i]['avg_agent_actor_grad_norm'].iloc[ep] if training_list[i][3][0] != 'test' else 0, 'Drone Type': 'Agent', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		df2 = pd.DataFrame([{'Average Actor Gradient Norm': metrics_list[i]['avg_adver_actor_grad_norm'].iloc[ep] if training_list[i][3][1] != 'test' else 0, 'Drone Type': 'Adver', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		data = pd.concat([data, df1, df2])
	
	# plot and save figure
	hue = data['Algorithms'].astype(str) + ', ' + data['Drone Type'].astype(str)
	plot = sns.lineplot(x = "Number of Episodes", y = "Average Actor Gradient Norm", hue = hue, data = data)
	plot.set_title("Average Actor Gradient Norm vs Number of Episodes")
	plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_actor_grad_norm_vs_number_of_episodes_lineplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_critic_grad_norm_vs_algorithms barplot

	# obtain data
	df1 = pd.DataFrame([{'Average Critic Gradient Norm': metrics_list[i]['avg_agent_critic_grad_norm'].iloc[number_of_episodes] if training_list[i][3][0] != 'test' else 0, 'Drone Type': 'Agent', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	df2 = pd.DataFrame([{'Average Critic Gradient Norm': metrics_list[i]['avg_adver_critic_grad_norm'].iloc[number_of_episodes] if training_list[i][3][1] != 'test' else 0, 'Drone Type': 'Adver', 
						 'Algorithms': vs_labels_list[i]} for i in range(len(metrics_list))])
	data = pd.concat([df1, df2])

	# plot and save figure
	plot = sns.barplot(x = "Algorithms", y = "Average Critic Gradient Norm", hue = "Drone Type", data = data)
	plot.set_title("Average Critic Gradient Norm vs Algorithms")
	plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_critic_grad_norm_vs_algorithms_barplot.pdf", bbox_inches = 'tight')
	plt.close()

	# average_critic_grad_norm_vs_number_of_episodes lineplot

	# obtain data 
	data = pd.DataFrame()

	for i in range(len(metrics_list)):

		df1 = pd.DataFrame([{'Average Critic Gradient Norm': metrics_list[i]['avg_agent_critic_grad_norm'].iloc[ep] if training_list[i][3][0] != 'test' else 0, 'Drone Type': 'Agent', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		df2 = pd.DataFrame([{'Average Critic Gradient Norm': metrics_list[i]['avg_adver_critic_grad_norm'].iloc[ep] if training_list[i][3][1] != 'test' else 0, 'Drone Type': 'Adver', 
							 'Number of Episodes': metrics_list[i]['episodes'].iloc[ep], 'Algorithms': vs_labels_list[i]} for ep in range(number_of_episodes)])
		data = pd.concat([data, df1, df2])
	
	# plot and save figure
	hue = data['Algorithms'].astype(str) + ', ' + data['Drone Type'].astype(str)
	plot = sns.lineplot(x = "Number of Episodes", y = "Average Critic Gradient Norm", hue = hue, data = data)
	plot.set_title("Average Critic Gradient Norm vs Number of Episodes")
	plt.legend(bbox_to_anchor = (1.01, 1), loc = 2)
	plt.savefig("training_plots/" + postprocessing_directory_name + "/average_critic_grad_norm_vs_number_of_episodes_lineplot.pdf", bbox_inches = 'tight')
	plt.close()


if __name__ == "__main__":

	# conduct the post processing for the case when this script is called manually
	post_process(number_of_episodes = NUMBER_OF_EPISODES, csv_log_directory = CSV_LOG_DIRECTORY, postprocessing_directory_name = POSTPROCESSING_DIRECTORY_NAME, training_list = TRAINING_LIST)	