class IrisInstance
  attr_accessor :sepal_length, :sepal_width, :petal_length, :petal_width, :given_label, :classified_class

  def initialize(sepal_length, sepal_width, petal_length, petal_width, given_label, classified_class = nil)
    self.sepal_length = sepal_length
    self.sepal_width  = sepal_width
    self.petal_length = petal_length
    self.petal_width  = petal_width
    @given_label      = given_label
    @classified_class = classified_class
  end

  def classified_class_mismatched?
    if given_label.nil? || classified_class.nil? # label not given, or not classified yet
      false
    else
      given_label != classified_class
    end
  end

  def given_label
    @given_label.nil? ? "Unlabeled" : @given_label
  end

  def classified_class
    @classified_class.nil? ? "Unclassified" : @classified_class
  end

  def to_s
    "[#{sepal_length}, #{sepal_width}, #{petal_length}, #{petal_width}, #{given_label}, #{classified_class}]"
  end

  def clone
    IrisInstance.new(sepal_length, sepal_width, petal_length, petal_width, @given_label, @classified_class)
  end
end
