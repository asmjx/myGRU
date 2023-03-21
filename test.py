import matplotlib.pyplot as plt    
def paint(self,log:OrderedDict):
        log['train_err'] = []
        log['test_err'] = []
        log['train_loss'] = []
        log['test_loss'] = []
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)

        ax1 = fig1.subplots() 
        ax2 = fig2.subplots()

        ax1.plot(log['train_err'], log['test_err'], label="err")
        ax2.plot(log['train_loss'], log['test_loss'], label="loss")

        ax1.set_title("err")
        ax2.set_title("loss")

        ax1.legend()
        ax2.legend()

        # plt.show()

        fig1.savefig("fig1.png")
        fig2.savefig("fig2.png")