package qlearning.util;

public class Tuple <K, V, E> {

    private final K key;
    private final V value;
    private final E element;

    public Tuple(K key, V value, E element) {
        this.key = key;
        this.value = value;
        this.element = element;
    }

    public K getItem1() {
        return key;
    }

    public V getItem2() {
        return value;
    }

    public E getItem3() {
        return element;
    }

    public String toString() {
        return "(" + key.toString() + ", " + value.toString() + ", " + element.toString() + ")";
    }

}
