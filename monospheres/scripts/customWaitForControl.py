#!/usr/bin/python3

import rospy
import rosnode
import time
import os
import rostopic
import rosgraph.masterapi
import random
import string

class TopicChecker:

    def __init__(self, topic):

        self.topic = topic

        self.sub = rospy.Subscriber(topic, rospy.AnyMsg, self.callback)

        self.got = False

    def callback(self, msg):
        self.got = True

    def ready(self):
        return self.got

    def getTopic(self):
        return self.topic

class Node:

    def checkMaster(self):
        try:
            rosgraph.masterapi.Master('/rostopic').getPid()
        except:
            return False

        return True

    def __init__(self):
        UAV_NAME = os.getenv('UAV_NAME')

        while not self.checkMaster():
            print("waiting for Master")
            time.sleep(1.0)

        try:
            random_str = str(''.join(random.choices(string.ascii_uppercase + string.digits, k=8)))
            rospy.init_node("waitForControl_{}_{}".format(UAV_NAME, random_str), anonymous=True)
        except:
            return
            pass

        topics = [
            "/" + UAV_NAME + "/control_manager/diagnostics",
            "/" + UAV_NAME + "/uav_manager/diagnostics",
            "/" + UAV_NAME + "/gain_manager/diagnostics",
            "/" + UAV_NAME + "/constraint_manager/diagnostics",
            "/" + UAV_NAME + "/estimation_manager/diagnostics",
            "/" + UAV_NAME + "/estimation_manager/uav_state",
        ]

        checkers = []

        for topic in topics:
            checkers.append(TopicChecker(topic))

        while not rospy.is_shutdown():

            ready = True

            for checker in checkers:
                if not checker.ready():
                    ready = False
                    print("waiting for {}".format(checker.getTopic()))
                    break

            if ready:
                break

            time.sleep(1.0)

if __name__ == '__main__':
    try:
        node = Node()
    except rospy.ROSInterruptException:
        pass
