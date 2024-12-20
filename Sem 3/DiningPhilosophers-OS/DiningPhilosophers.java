import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

class Philosopher extends Thread {
    private final int id;
    private final Lock leftFork;
    private final Lock rightFork;

    public Philosopher(int id, Lock leftFork, Lock rightFork) {
        this.id = id;
        this.leftFork = leftFork;
        this.rightFork = rightFork;
    }

    private void eat() throws InterruptedException {
        System.out.println("Philosopher " + id + " is eating.");
        Thread.sleep((long) (Math.random() * 1000));
    }

    private void get() throws InterruptedException {
        System.out.println("Philosopher " + id + " get the forks.");
    }

    private void put() throws InterruptedException {
        System.out.println("Philosopher " + id + " put down the forks.");
    }

    private void thinking() throws InterruptedException {
        Thread.sleep((long) (Math.random() * 1000));
    }

    @Override
    public void run() {
        try {
            while (true) {
                thinking();  // Adding thinking time before trying to pick up forks

                // Pick up forks (lower-numbered first)
                Lock firstFork = leftFork;
                Lock secondFork = rightFork;

                if (leftFork.hashCode() > rightFork.hashCode()) {
                    firstFork = rightFork;
                    secondFork = leftFork;
                }

                firstFork.lock();
                secondFork.lock();
                get();

                eat();

                // Put down forks
                put();
                firstFork.unlock();
                secondFork.unlock();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}

public class DiningPhilosophers {
    private static final int NUM_PHILOSOPHERS = 5;

    public static void main(String[] args) {
        Lock[] forks = new ReentrantLock[NUM_PHILOSOPHERS];
        Philosopher[] philosophers = new Philosopher[NUM_PHILOSOPHERS];

        for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
            forks[i] = new ReentrantLock();
        }

        for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
            Lock leftFork = forks[i];
            Lock rightFork = forks[(i + 1) % NUM_PHILOSOPHERS];
            philosophers[i] = new Philosopher(i, leftFork, rightFork);
            philosophers[i].start();
        }

        // Let the simulation run for a while
        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // Stop all philosophers
        for (Philosopher philosopher : philosophers) {
            philosopher.interrupt();
        }
    }
}